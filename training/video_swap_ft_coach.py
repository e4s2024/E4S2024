import os
import cv2
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
from datasets.video_swap_dataset import VideoFaceSwappingDataset
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
from swap_face_fine.face_parsing.face_parsing_demo import  vis_parsing_maps


def dialate_mask(mask, img, radius=3, verbose=False):
    """
        将mask dialate一下
        
        输入的mask必须是12个类别的, img是 PIL 图片 (1024分辨率)
    """
    
    # 找到face region
    hair_bg_mask = np.stack([np.equal(mask, clz) for clz in [0,4,7,8,11]], axis=0).any(axis=0)
    face_mask = np.logical_not(hair_bg_mask)
    
    # dilate 一下
    kernel_size = (radius * 2 + 1, radius * 2 + 1)
    kernel = np.ones(kernel_size)
    dilated_face_mask = cv2.dilate((255*face_mask).astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    
    dilated_mask = np.zeros_like(mask)
    dilated_mask[np.equal(dilated_face_mask, 255)] = mask[np.equal(dilated_face_mask, 255)]
    
    dilated_region = dilated_face_mask - (255*face_mask).astype(np.uint8)
    dilated_mask[np.equal(dilated_region, 255)] = 6 # 外扩区域填皮肤
    
    # TODO 可视化一下 erode mask 和 原始的 mask
    if verbose:
        orig_mask_vis = vis_parsing_maps(img, mask)
        dilated_mask_vis = vis_parsing_maps(img, dilated_mask)
        comp = Image.fromarray(np.hstack([orig_mask_vis, dilated_mask_vis]))
        return dilated_mask, comp
        
    return dilated_mask, None

def erode_mask(mask, img, radius=3, verbose=False):
    """
        将mask erode一下
        
        输入的mask必须是12个类别的, img是 PIL 图片 (1024分辨率)
    """
    
    # # 找到face region
    hair_bg_mask = np.stack([np.equal(mask, clz) for clz in [0,4,11]], axis=0).any(axis=0)
    face_mask = np.logical_not(hair_bg_mask)
    
    # 找到face region
    # face_mask = np.equal(mask, 6) 
    
    # erode 一下
    kernel_size = (radius * 2 + 1, radius * 2 + 1)
    kernel = np.ones(kernel_size)
    eroded_face_mask = cv2.erode((255*face_mask).astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    
    eroded_mask = np.zeros_like(mask)
    eroded_mask[np.equal(eroded_face_mask, 255)] = mask[np.equal(eroded_face_mask, 255)]
    
    # TODO 可视化一下 erode mask 和 原始的 mask
    if verbose:
        orig_mask_vis = vis_parsing_maps(img, mask)
        eroded_mask_vis = vis_parsing_maps(img, eroded_mask)
        comp = Image.fromarray(np.hstack([orig_mask_vis, eroded_mask_vis]))
        return eroded_mask, comp
        
    return eroded_mask, None


def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)

def logical_and_reduce(*tensors):
    return torch.stack(tensors, dim=0).all(dim=0)


class VideoSwapPTICoach:
    def __init__(self, opts, e4s_net=None, num_targets=50, erode=False,
                 ):
        self.opts = opts

        self.erode = erode
        self.device = torch.device("cuda", 0)
        # self.opts.device = self.device
        
        # 定义数据集
        self.dataset = self.configure_dataset(num_targets)
        if num_targets == -1:
            num_targets = len(self.dataset)
        self.num_targets = num_targets
        
        # 定义 loss function
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = IDLoss(self.opts).to(self.device).eval()
        if self.opts.face_parsing_lambda > 0:
            self.face_parsing_loss = FaceParsingLoss(self.opts).to(self.device).eval()
    
        # 初始化网络
        if e4s_net is None:
            self.net = Net3(self.opts)
            # print(self.device)
            self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self.net = self.net.to(self.device)
        else:
            self.net = e4s_net
        
        # 加载整个模型预训练好的参数，作为初始化
        assert self.opts.checkpoint_path is not None, "必须提供预训练好的参数!"
        ckpt_dict = torch.load(self.opts.checkpoint_path)
        self.net.latent_avg = ckpt_dict['latent_avg'].to(self.device)
        self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
        print("Load pre-trained model success!")        
        
        # 初始化优化器
        self.optimizer = self.configure_optimizer()

        # # 保存优化后模型的地址
        # self.checkpoint_dir = os.path.join(self.opts.exp_dir, 'checkpoints')
        # os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize tensorborad logger
        self.log_dir = os.path.join(self.opts.exp_dir, 'logs_lr%f_iters%d_erode%d_run2'%(self.opts.pti_learning_rate, self.opts.max_pti_steps, self.opts.erode_radius))
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = SummaryWriter(logdir = self.log_dir)
        
    def configure_dataset(self, num_targets: int = -1):
        save_dir = self.opts.exp_dir
        ds = VideoFaceSwappingDataset(
            driven = sorted(glob.glob(os.path.join(save_dir,"imgs", "D_*.png")))[:num_targets],
            driven_recolor = sorted(glob.glob(os.path.join(save_dir, "imgs", "D_recolor_*.png")))[:num_targets],
            driven_mask = sorted(glob.glob(os.path.join(save_dir,"mask","D_mask_*.png")))[:num_targets],
            driven_style_vector = sorted(glob.glob(os.path.join(save_dir,"styleVec","D_style_vec_*.pt")))[:num_targets],
            target = sorted(glob.glob(os.path.join(save_dir,"imgs", "T_*.png")))[:num_targets],
            target_mask = sorted(glob.glob(os.path.join(save_dir,"mask","T_mask_*.png")))[:num_targets],
            target_style_vector = sorted(glob.glob(os.path.join(save_dir,"styleVec","T_style_vec_*.pt")))[:num_targets],
            img_transform=transforms.Compose([TO_TENSOR, NORMALIZE]),
            label_transform=transforms.Compose([TO_TENSOR])
        ) 
        
        return ds
        
    def configure_optimizer(self):
        self.params = list(filter(lambda p: p.requires_grad ,list(self.net.parameters())))
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(self.params, lr=self.opts.pti_learning_rate)
        else:
            optimizer = Ranger(self.params, lr=self.opts.pti_learning_rate)
        return optimizer
        
    def calc_loss(self, img, img_recon, foreground_mask=None):
        """
            img: target 图片
            img_recon: 当前得到的结果 
        """
        loss_dict = {}
        loss = 0.0
        id_logs = None
        
        if foreground_mask is not None:
            img_recon = img_recon * foreground_mask
            img = img * foreground_mask 
                            
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
    
    @torch.no_grad()
    def recon_driven(self):
        self.net.eval()

        print('Reconstrcution driven videos...')
        for idx, (driven_image, driven_m, driven_s_v,
                  target_image, target_m, target_s_v,
                  driven_recolor_pil, driven_pil, target_pil) in tqdm(enumerate(self.dataset)): # 从idx = 0 开始
        
            driven_m = (driven_m*255).long().to(self.opts.device).unsqueeze(0)
            driven_onehot = torch_utils.labelMap2OneHot(driven_m, num_cls=self.opts.num_seg_cls)
            driven_style_vector = driven_s_v.to(self.opts.device).float()
            driven_style_code = self.net.cal_style_codes(driven_style_vector)
            
            recon_i, _, structure_feats_i  = self.net.gen_img(torch.zeros(1,512,32,32).to(self.opts.device), driven_style_code, driven_onehot)
                                                            #   randomize_noise=False,noise=noise)
            torch_utils.tensor2im(recon_i[0]).save(os.path.join(self.opts.exp_dir, "imgs", "D_finetuned_recon_%04d.png"%idx))

    def train(self):
        self.train_e4s()

    def train_e4s(self):
        self.net.train()
        
        print('Fine tuning the network...')
        for step in trange(self.opts.max_pti_steps):
            step_loss_dict = defaultdict(list)
            t = (step + 1) / self.opts.max_pti_steps

            verbose_recon = None
            for idx, (driven_image, driven_m, driven_s_v,
                      target_image, target_m, target_s_v,
                      driven_recolor_pil, driven_pil, target_pil) in enumerate(tqdm(self.dataset,
                                                    desc=f"tuning e4s_g {step}/{self.opts.max_pti_steps}",
                                                    position=0,
                                                    )):  # 从idx = 0 开始
                driven_m = (driven_m*255).long().to(self.opts.device).unsqueeze(0)

                if self.erode:
                    driven_pil = Image.fromarray(np.transpose((255*(driven_image.numpy()+1)/2).astype(np.uint8), (1,2,0)))
                    driven_m_np = driven_m[0,0,:,:].cpu().numpy().astype(np.uint8)
                    driven_m_eroded, erode_verbose = erode_mask(driven_m_np, driven_pil , radius=self.opts.erode_radius, verbose=True)
                    driven_m = torch.from_numpy(driven_m_eroded).long().to(self.opts.device).unsqueeze(0).unsqueeze(0)

                driven_image = driven_image.to(self.opts.device).float().unsqueeze(0)
                driven_onehot = torch_utils.labelMap2OneHot(driven_m, num_cls=self.opts.num_seg_cls)
                driven_style_vector = driven_s_v.to(self.opts.device).float()
                driven_style_code = self.net.cal_style_codes(driven_style_vector)

                zero_latent = torch.zeros((1,512,32,32), requires_grad=False).to(self.opts.device)
                recon_i, _, structure_feats_i  = self.net.gen_img(zero_latent, driven_style_code, driven_onehot)
                                                                #   in [-1,1]

                ''' also guided by recolor net '''
                recolor_i = torch_utils.im2tensor(driven_recolor_pil, std=False)

                mask_bg_and_hair = logical_or_reduce(*[driven_m == clz for clz in [0, 4, 11]])
                is_foreground = torch.logical_not(mask_bg_and_hair)
                foreground_mask = is_foreground.float()
                foreground_mask = F.interpolate(foreground_mask, (1024, 1024), mode='bilinear', align_corners=False)
                if self.erode:
                    loss, loss_dict, id_logs = self.calc_loss(driven_image, recon_i, foreground_mask=foreground_mask)
                else:
                    loss, loss_dict, id_logs = self.calc_loss(driven_image, recon_i)

                loss_recolor, _, _ = self.calc_loss(recolor_i, recon_i, foreground_mask=foreground_mask)
                loss += loss_recolor * self.opts.recolor_lambda  # default: 0.5?
                
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
            
            if step+1 == 100:
                save_dict = self.get_save_dict()
                torch.save(save_dict, os.path.join(self.opts.exp_dir, "finetuned_G_lr%f_iters%d.pth"%(self.opts.pti_learning_rate, step+1)))
        
        print('Finished fine-tuning e4s generator!')

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

    def freeze_e4s_g(self):
        self.net.requires_grad_(False)
