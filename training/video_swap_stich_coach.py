import os
import os.path
from collections import defaultdict
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm, trange

# 自己加的
from PIL import Image
import glob
from datasets.dataset import  TO_TENSOR, NORMALIZE, MASK_CONVERT_TF, FFHQDataset, FFHQ_MASK_CONVERT_TF, MASK_CONVERT_TF_DETAILED, FFHQ_MASK_CONVERT_TF_DETAILED
from datasets.video_swap_dataset import VideoFaceSwappingStichingDataset
from criteria.id_loss import IDLoss
from criteria.face_parsing.face_parsing_loss import FaceParsingLoss
from criteria.lpips.lpips import LPIPS
from criteria.style_loss import StyleLoss
from models.networks import Net3
from training.ranger import Ranger
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
from utils import torch_utils
import math
from collections import OrderedDict
from tensorboardX import SummaryWriter
import copy
from utils.morphology import dilation

def create_masks(mask, outer_dilation=0):
    temp = copy.deepcopy(mask)
    full_mask = dilation(temp, torch.ones(2 * outer_dilation + 1, 2 * outer_dilation + 1, device=mask.device), engine='convolution')
    border_mask = full_mask - temp

    border_mask = border_mask.clip(0, 1)
    content_mask = mask
    
    return content_mask, border_mask, full_mask

def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)

def logical_and_reduce(*tensors):
    return torch.stack(tensors, dim=0).all(dim=0)

class VideoSwapStichingCoach:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device("cuda", 0)
        # self.opts.device = self.device
        
        # 定义数据集
        self.dataset = self.configure_dataset()
        
        # 定义 loss function
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = IDLoss(self.opts).to(self.device).eval()
        if self.opts.face_parsing_lambda > 0:
            self.face_parsing_loss = FaceParsingLoss(self.opts).to(self.device).eval()
    
        # 初始化网络
        self.net = Net3(self.opts)
        # print(self.device)
        # self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net = self.net.to(self.device) 
        
        # 加载整个模型预训练好的参数，作为初始化
        assert self.opts.PTI_checkpoint_path is not None, "必须提供预训练好的参数!"
        ckpt_dict = torch.load(self.opts.PTI_checkpoint_path)
        self.net.latent_avg = ckpt_dict['latent_avg'].to(self.device)
        self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"], prefix="module."))
        print("Load PTI finetuned model success!")      
        
        # 初始化优化器
        self.optimizer = self.configure_optimizer()

        # # 保存优化后模型的地址
        # self.checkpoint_dir = os.path.join(self.opts.exp_dir, 'checkpoints')
        # os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize tensorborad logger
        self.log_dir = os.path.join(self.opts.exp_dir, 'logs_lr1e2_iters100_stiching')
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = SummaryWriter(logdir = self.log_dir)
        
    def configure_dataset(self):  
        save_dir = os.path.join(self.opts.exp_dir, "intermediate_results")
        ds = VideoFaceSwappingStichingDataset(
            swapped_mask = sorted(glob.glob(os.path.join(save_dir,"mask", "swappedMask_*.png"))),  
            swapped_style_vector = sorted(glob.glob(os.path.join(save_dir,"styleVec","swapped_style_vec_*.npy"))),
            
            content_img = sorted(glob.glob(os.path.join(save_dir,"imgs", "Swapped_after_PTI_*.png"))), 
            border_img  = sorted(glob.glob(os.path.join(save_dir,"imgs", "T_*.png"))), 
            
            img_transform = transforms.Compose([TO_TENSOR, NORMALIZE]),
            label_transform = transforms.Compose([TO_TENSOR])
        ) 
        
        return ds
        
    def configure_optimizer(self):
        self.params=list(filter(lambda p: p.requires_grad ,list(self.net.parameters())))
        
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(self.params, lr=self.opts.stiching_learning_rate)
        else:
            optimizer = Ranger(self.params, lr=self.opts.stiching_learning_rate)
            
        return optimizer
    
    
    def calc_loss(self, img, img_recon):
        """
            img: target 图片
            img_recon: 当前得到的结果 
        """
        loss_dict = {}
        loss = 0.0
        id_logs = None
                            
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(img_recon, img)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(img_recon, img)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = 0
            for i in range(3):
                loss_lpips_1 = self.lpips_loss(
                    F.adaptive_avg_pool2d(img_recon,(1024//2**i,1024//2**i)), 
                    F.adaptive_avg_pool2d(img,(1024//2**i,1024//2**i))
                )
                loss_lpips += loss_lpips_1
            
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.face_parsing_lambda > 0:
            loss_face_parsing, face_parsing_sim_improvement = self.face_parsing_loss(img_recon, img)
            loss_dict['loss_face_parsing'] = float(loss_face_parsing)
            loss_dict['face_parsing_improve'] = float(face_parsing_sim_improvement)
            loss += loss_face_parsing * self.opts.face_parsing_lambda
       
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs
    
    def calc_loss_withBorderMask(self, recon_i, border_img, border_mask, content_img, content_mask):
        # 换脸的loss，一张target, 一张 driven，同时约束生成的结果
        """
            
            recon_i: 当前得到的结果 
            border_img: 提供 border的图片，一般是target
            content_img: 提供face的图片，一般是 finetune 后的模型在交换 mask后的图片
            face_mask: 脸部区域的mask
            border_mask: 脸部dialation出来的边界区域
        """
        loss_dict = {}
        loss = 0.0
        id_logs = None
        
        
        boder_region = border_img * border_mask    # dialation 区域
        content_region = content_img * content_mask   # 驱动结果的 face 区域   
        
        recon_boder_region = recon_i * border_mask  # 当前结果的 dialation 区域
        recon_content_region = recon_i * content_mask  # 当前结果的face区域
                            
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(recon_content_region, content_region)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(recon_content_region, content_region) + F.mse_loss(recon_boder_region, boder_region)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = 0
            for i in range(3):
                loss_lpips_1 = self.lpips_loss(
                    F.adaptive_avg_pool2d(recon_content_region,(1024//2**i,1024//2**i)), 
                    F.adaptive_avg_pool2d(content_region,(1024//2**i,1024//2**i))
                )
                loss_lpips_2 = self.lpips_loss(
                    F.adaptive_avg_pool2d(recon_boder_region,(1024//2**i,1024//2**i)), 
                    F.adaptive_avg_pool2d(boder_region,(1024//2**i,1024//2**i))
                )
                loss_lpips += (loss_lpips_1 + loss_lpips_2)
            
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.face_parsing_lambda > 0:
            loss_face_parsing, face_parsing_sim_improvement = self.face_parsing_loss(recon_content_region, content_region)
            loss_dict['loss_face_parsing'] = float(loss_face_parsing)
            loss_dict['face_parsing_improve'] = float(face_parsing_sim_improvement)
            loss += loss_face_parsing * self.opts.face_parsing_lambda
       
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs
        
    # @torch.no_grad()
    def train(self):
        self.net.train() 
        
        print('Fine tuning the network when stiching...')
        for step in trange(self.opts.max_stiching_steps):
            step_loss_dict = defaultdict(list)
            t = (step + 1) / self.opts.max_stiching_steps

            verbose_recon = None
            for idx, (content_image, border_image, swapped_m, swapped_s_v) in tqdm(enumerate(self.dataset)): # 从idx = 0 开始
           
                content_image = content_image.to(self.opts.device).float().unsqueeze(0)
                border_image = border_image.to(self.opts.device).float().unsqueeze(0)
                swapped_m = (swapped_m*255).long().to(self.opts.device).unsqueeze(0)
                swapped_onehot = torch_utils.labelMap2OneHot(swapped_m, num_cls=self.opts.num_seg_cls)
                
                swapped_style_vector = swapped_s_v.to(self.opts.device).float()
                swapped_style_code = self.net.cal_style_codes(swapped_style_vector)
                
                recon_i, _, structure_feats_i  = self.net.gen_img(torch.zeros(1,512,32,32).to(self.opts.device), swapped_style_code, swapped_onehot)
                                                                #   randomize_noise=False,noise=noise)
                
                # 处理一下mask， 得到face 区域和 dialation区域
                mask_bg_and_hair = logical_or_reduce(*[swapped_m == clz for clz in [0,4,11]])
                is_foreground = torch.logical_not(mask_bg_and_hair)
                foreground_mask = is_foreground.float()
                
                content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation = self.opts.outer_dilation)
        
                content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
                border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
                full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=False)
                
                # loss, loss_dict, id_logs = self.calc_loss(content_image, recon_i)
                loss, loss_dict, id_logs = self.calc_loss_withBorderMask(recon_i, border_image, border_mask, content_image, content_mask)
                
                if idx == 0:
                    verbose_recon = np.array(torch_utils.tensor2im(recon_i[0]))
                
                
                step_loss_dict['loss'].append(loss.item())
                for k,v in loss_dict.items():
                    if "loss_" in k:
                        step_loss_dict[k].append(v)
                                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 记录视频序列中第一张图片在每个step后的结果
            self.logger.add_image("image_recon", verbose_recon, step, dataformats='HWC')

            # 记录 每个step 视频中所有帧的 平均loss
            log_dict = {}
            for key, losses in step_loss_dict.items():
                loss_mean = sum(losses) / len(losses)
                loss_max = max(losses)
                
                self.logger.add_scalar(f'loss_mean/{key}', loss_mean, step)
                self.logger.add_scalar(f'loss_max/{key}', loss_max, step)
        
        print('Finished stiching fine-tuning!')
        
        
    def checkpoint_me(self):
        save_name = 'finetuned_model_%d.pt'%self.opts.max_pti_steps
        save_dict = self.get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        
    def get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts),
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] =  self.net.latent_avg
            
        return save_dict