import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.dataset import CelebAHQDataset, get_transforms, TO_TENSOR, NORMALIZE, MASK_CONVERT_TF, MASK_CONVERT_TF_DETAILED, FFHQ_MASK_CONVERT_TF_DETAILED
from models.networks import Net3
from options.optim_options import OptimOptions
from criteria.id_loss import IDLoss
from criteria.lpips.lpips import LPIPS
from criteria.style_loss import StyleLoss
from criteria.face_parsing.face_parsing_loss import FaceParsingLoss
import os
import json
import sys
import pprint
import torch
from functools import partial
from utils import torch_utils
import random
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import make_grid
import torch.nn as nn
import glob
from PIL import Image
from utils.alignment import crop_faces, calc_alignment_coefficients
from utils.morphology import dilation

from swap_face_fine.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps

sys.path.append(".")
sys.path.append("..")

toPIL = transforms.ToPILImage()

celelbAHQ_label_list = ['background','skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']


# supported_swap_comps= ['background','skin', 'nose', 'eye',
#                         'brow','ear', 'mouth','hair', 
#                         'hat','ear_r','neck', 'cloth']

# 9个属性
faceParser_label_list = ['background', 'mouth', 'eyebrows',
                         'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface']

def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image

def save_image(image, output_folder, image_name, image_index, ext='jpg'):
    if ext == 'jpeg' or ext == 'jpg':
        image = image.convert('RGB')
    folder = os.path.join(output_folder, image_name)
    os.makedirs(folder, exist_ok=True)
    image.save(os.path.join(folder, "%04d.%s"%(image_index,ext)))
    
class Optimizer:
    def __init__(self, opts, net=None):
        self.opts = opts
        
        if net is None:
            """
            self.test_ds = CelebAHQDataset(dataset_root=self.opts.dataset_root, mode="test",
                                        img_transform=transforms.Compose(
                                            [TO_TENSOR, NORMALIZE]),
                                        label_transform=transforms.Compose(
                                            [ MASK_CONVERT_TF_DETAILED,TO_TENSOR]), # MASK_CONVERT_TF,
                                        fraction=self.opts.ds_frac)
            print(f"Number of test samples: {len(self.test_ds)}")
            """

            # self.test_dataloader = DataLoader(self.test_ds, batch_size=self.opts.test_batch_size,
            #                                   shuffle=False, num_workers=int(self.opts.test_workers), drop_last=False)
        
        
            assert self.opts.checkpoint_path is not None, "please specify the pre-trained weights!"
            self.net = Net3(self.opts).eval().to(self.opts.device)
        
            ckpt_dict = torch.load(self.opts.checkpoint_path)
            self.net.latent_avg = ckpt_dict['latent_avg'].to(self.opts.device) if self.opts.start_from_latent_avg else None
            if self.opts.load_ema:
                self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict_ema"],prefix="module."))
            else:            
                self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
            
            print("Load pre-trained weights.")
        
            # # 重新保存一下
            # torch.save(ckpt_dict,"./ckpt.pth",_use_new_zipfile_serialization=False)
        else:
            self.net = net
        
        # loss 函数
        self.mse_loss = nn.MSELoss().to(self.opts.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.opts.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = IDLoss(self.opts).to(self.opts.device).eval()
        if self.opts.face_parsing_lambda > 0:
            self.face_parsing_loss = FaceParsingLoss(self.opts).to(self.opts.device).eval()
       
        self.img_transform = transforms.Compose([TO_TENSOR, NORMALIZE])
        self.label_transform_wo_converter = transforms.Compose([TO_TENSOR])
        self.label_transform_w_converter = transforms.Compose([MASK_CONVERT_TF_DETAILED, TO_TENSOR])
    
    def calc_loss(self, img, img_recon, mask):
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
        # if self.opts.style_lambda > 0:  # gram matrix loss
        #     loss_style = self.style_loss(img_recon, img, mask_x = (mask==4).float(), mask_x_hat = (mask==4).float())
        #     loss_dict['loss_style'] = float(loss_style)
        #     loss += loss_style * self.opts.style_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs
          
    def calc_loss_swappingFace(self, img, img_recon_i, img_recon, mask):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        
        # 先把 background, hair 选出来
        mask = F.interpolate(mask.float(),(1024,1024),mode='nearest').long()
        mask_bg_and_hair = torch.logical_or(mask == 0, mask == 4)
        mask_bg_and_hair = torch.logical_or(mask_bg_and_hair, mask == 7)
        mask_bg_and_hair = torch.logical_or(mask_bg_and_hair, mask == 8)
        
        img_bg_and_hair = img * mask_bg_and_hair    # target 的背景和头发区域
        img_rcon_face = img_recon * torch.logical_not(mask_bg_and_hair)    # 一阶段的face区域   
        img_recon_i_bg_and_hair = img_recon_i * mask_bg_and_hair  # 当前结果的背景和头发区域
        img_recon_i_face = img_recon_i * torch.logical_not(mask_bg_and_hair)  # 当前结果的face区域
                            
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(img_recon_i_face, img_rcon_face)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(img_recon_i_bg_and_hair, img_bg_and_hair) + F.mse_loss(img_recon_i_face, img_rcon_face)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = 0
            for i in range(3):
                loss_lpips_1 = self.lpips_loss(
                    F.adaptive_avg_pool2d(img_recon_i_bg_and_hair,(1024//2**i,1024//2**i)), 
                    F.adaptive_avg_pool2d(img_bg_and_hair,(1024//2**i,1024//2**i))
                )
                loss_lpips_2 = self.lpips_loss(
                    F.adaptive_avg_pool2d(img_recon_i_face,(1024//2**i,1024//2**i)), 
                    F.adaptive_avg_pool2d(img_rcon_face,(1024//2**i,1024//2**i))
                )
                loss_lpips += (loss_lpips_1 + loss_lpips_2)
            
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.face_parsing_lambda > 0:
            loss_face_parsing, face_parsing_sim_improvement = self.face_parsing_loss(img_recon_i_face, img_rcon_face)
            loss_dict['loss_face_parsing'] = float(loss_face_parsing)
            loss_dict['face_parsing_improve'] = float(face_parsing_sim_improvement)
            loss += loss_face_parsing * self.opts.face_parsing_lambda
       
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs
    
    def setup_W_optimizer(self, W_init,noise_init=None):
        """设置好优化器

        Args:
            W_init (Tensor): W+ space 各个component的初始参数，with shape [bs,#seg_cls,18,512]

        Returns:
            [type]: [description]
        """

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        
        tmp = W_init.clone().detach()  # clone 一下 latent avg
        tmp.requires_grad = True

        params = [tmp]
        
        if noise_init is not None:
            noises = []
            for noise in noise_init:
                noise_cp = noise.clone().detach()
                noise_cp.requires_grad = True
                
                noises.append(noise_cp)
                
            params.extend(noises)  
            
        optimizer_W = opt_dict[self.opts.opt_name](params, lr=self.opts.lr)

        if noise_init is not None:
            return optimizer_W, tmp, noises
        else:
            return optimizer_W, tmp
    
    def inversion_img(self, img_path, mask_path):
        img_name = os.path.basename(img_path).split(".")[0]
        intermediate_folder = os.path.join(self.opts.output_dir, img_name)
        os.makedirs(intermediate_folder, exist_ok=True)

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        img, mask = self.img_transform(img), self.label_transform_wo_converter(mask)
      
        img, mask = img.unsqueeze(0), mask.unsqueeze(0)
        img = img.to(self.opts.device).float()
        mask = (mask * 255).long().to(self.opts.device)
        # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
        onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
        
        # 原始的style codes
        with torch.no_grad():
            # struc_code, style_codes = self.net.get_style(img, onehot)
            style_vectors, struc_code = self.net.get_style_vectors(img, onehot)
            style_codes = self.net.cal_style_codes(style_vectors)
        
            
        recon, _, structure_feats = self.net.gen_img(torch.zeros_like(struc_code), style_codes, onehot)
                                                    #  randomize_noise=False,noise=noise)
        
        img_vis = torch_utils.tensor2im(img[0])
        img_vis.save(os.path.join(self.opts.output_dir, img_name, img_name+"_gt.png"))
        recon_vis = torch_utils.tensor2im(recon[0])
        recon_vis.save(os.path.join(self.opts.output_dir, img_name, img_name+"_recon.png"))
        ## =======================================

        # Fine tuning area including eyes, eyebrows, lip, and mouth     
        finetune_mask = torch.logical_or(mask == 2, mask == 3)
        finetune_mask = torch.logical_or(finetune_mask, torch.logical_or(mask == 1, mask == 9)).float()
        finetune_mask = F.interpolate(finetune_mask, size=1024, mode='bilinear').clamp_(0, 1)
        finetune_mask = (finetune_mask > 0).float()
        # Fine tuning area dilation
        radius = 20
        finetune_mask = dilation(finetune_mask, torch.ones(2 * radius + 1, 2 * radius + 1, device=finetune_mask.device), engine='convolution')
        finetune_mask = dilation(finetune_mask, torch.ones(2 * radius + 1, 2 * radius + 1, device=finetune_mask.device), engine='convolution')
        finetune_mask = dilation(finetune_mask, torch.ones(2 * radius + 1, 2 * radius + 1, device=finetune_mask.device), engine='convolution')
        radius = 10
        finetune_mask = dilation(finetune_mask, torch.ones(2 * radius + 1, 2 * radius + 1, device=finetune_mask.device), engine='convolution')

        if self.opts.verbose:
            finetune_mask_image = transforms.ToPILImage()(finetune_mask[0].detach().cpu())
            finetune_mask_image.save(os.path.join(intermediate_folder, 'finetune_mask.png'))
        
        optimizer_W, latent = self.setup_W_optimizer(style_vectors,noise_init=None)
        pbar = tqdm(range(self.opts.W_steps), desc='Optimizing style code...', leave=False)
        for step in pbar:
            optimizer_W.zero_grad()
            
            style_codes = self.net.cal_style_codes(latent)
            
            recon_i, _, structure_feats_i  = self.net.gen_img(torch.zeros_like(struc_code), style_codes, onehot)
                                                            #   randomize_noise=False,noise=noise)
            
            # recon_i = recon
            '''
            loss_1, loss_dict, id_logs = self.calc_loss(img, recon_i * finetune_mask + img * (1 - finetune_mask), mask)
            loss_2, loss_dict, id_logs = self.calc_loss(img, img * finetune_mask + recon_i * (1 - finetune_mask), mask)

            print('loss_1: {}'.format(loss_1))
            print('loss_2: {}'.format(loss_2))
            loss = loss_1 * 500 + loss_2
            '''

            loss, loss_dict, id_logs = self.calc_loss(img, recon_i, mask)
            
            loss.backward()
            optimizer_W.step()  

            if self.opts.verbose:
                verbose_str = "[%s]"%img_name
                for k,v in loss_dict.items():
                    if k[:4]=="loss":
                        verbose_str += (k + " : {:.4f}, ".format(v))
                pbar.set_description(verbose_str)

            if self.opts.save_intermediate and (step+1) % self.opts.save_interval== 0:
                self.save_W_intermediate_results(img_name, recon_i, latent, step+1)
        
        self.save_W_intermediate_results(img_name, recon_i, latent, self.opts.W_steps, noise=None)

    def optim_W_online(self, img, mask):
        img, mask = self.img_transform(img), self.label_transform_wo_converter(mask)
      
        img, mask = img.unsqueeze(0), mask.unsqueeze(0)
        img = img.to(self.opts.device).float()
        mask = (mask * 255).long().to(self.opts.device)
        # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
        onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
        
        with torch.no_grad():
            # struc_code, style_codes = self.net.get_style(img, onehot)
            style_vectors, struc_code = self.net.get_style_vectors(img, onehot)
            # style_codes = self.net.cal_style_codes(style_vectors)

        optimizer_W, latent = self.setup_W_optimizer(style_vectors, noise_init=None)
        pbar = tqdm(range(self.opts.W_steps), desc='Optimizing style code...', leave=False)
        for step in pbar:
            optimizer_W.zero_grad()
            
            style_codes = self.net.cal_style_codes(latent)
            
            recon_i, _, _  = self.net.gen_img(torch.zeros_like(struc_code), style_codes, onehot)

            loss, loss_dict, id_logs = self.calc_loss(img, recon_i, mask)
            
            loss.backward()
            optimizer_W.step()  

        return latent.detach()


    def finetune_net(self, target_path, mask_path, style_vectors_path):
        self.net.train()

        checkpoint_dir = os.path.join(self.opts.exp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        img_name = os.path.basename(target_path).split(".")[0]

        target = Image.open(target_path)
        mask = Image.open(mask_path)
        style_vectors = np.load(style_vectors_path)
        style_vectors = torch.from_numpy(style_vectors).to(self.opts.device)
        
        target, mask = self.img_transform(target), self.label_transform_wo_converter(mask)
      
        target, mask = target.unsqueeze(0), mask.unsqueeze(0)
        target = target.to(self.opts.device).float()
        mask = (mask * 255).long().to(self.opts.device)
        # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
        onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
        
        optimizer_net = self.configure_optimizer()

        pbar = tqdm(range(self.opts.finetune_steps), desc='Fine tuning the network...', leave=False)
        for step in pbar:
            optimizer_net.zero_grad()
            
            style_codes = self.net.cal_style_codes(style_vectors)
            
            recon_i, _, structure_feats_i  = self.net.gen_img(torch.zeros(1, 512, 32, 32), style_codes, onehot)
                                                            #   randomize_noise=False,noise=noise)
            
            # recon_i = recon
            '''
            loss_1, loss_dict, id_logs = self.calc_loss(img, recon_i * finetune_mask + img * (1 - finetune_mask), mask)
            loss_2, loss_dict, id_logs = self.calc_loss(img, img * finetune_mask + recon_i * (1 - finetune_mask), mask)

            print('loss_1: {}'.format(loss_1))
            print('loss_2: {}'.format(loss_2))
            loss = loss_1 * 500 + loss_2
            '''

            loss, loss_dict, id_logs = self.calc_loss(target, recon_i, mask)
            
            loss.backward()
            optimizer_net.step()  

            if self.opts.verbose:
                verbose_str = "[finetune]"
                for k, v in loss_dict.items():
                    if k[:4]=="loss":
                        verbose_str += (k + " : {:.4f}, ".format(v))
                pbar.set_description(verbose_str)

            if self.opts.save_intermediate and (step+1) % self.opts.save_interval== 0:
                save_im = toPIL(((recon_i[0] + 1) / 2).detach().cpu().clamp(0, 1))
                # os.makedirs(intermediate_folder, exist_ok=True)
                image_path = os.path.join(self.opts.exp_dir, f'{img_name}_{step:04}.png')
                save_im.save(image_path)

        save_name = 'finetuned_model.pt'
        checkpoint_path = os.path.join(checkpoint_dir, save_name)
        save_dict = {
            'state_dict': self.net.state_dict(),
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] =  self.net.latent_avg
        torch.save(save_dict, checkpoint_path)


    def finetune_net_online(self, target, mask, style_vectors):
        self.net.train()

        # target = Image.open(target_path)
        # mask = Image.open(mask_path)
        # style_vectors = np.load(style_vectors_path)
        # style_vectors = torch.from_numpy(style_vectors).to(self.opts.device)
        
        target, mask = self.img_transform(target), self.label_transform_wo_converter(mask)
      
        target, mask = target.unsqueeze(0), mask.unsqueeze(0)
        target = target.to(self.opts.device).float()
        mask = (mask * 255).long().to(self.opts.device)
        # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
        onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
        
        optimizer_net = self.configure_optimizer()

        pbar = tqdm(range(self.opts.finetune_steps), desc='Fine tuning the network...', leave=False)
        for step in pbar:
            optimizer_net.zero_grad()
            
            style_codes = self.net.cal_style_codes(style_vectors)
            
            recon_i, _, structure_feats_i  = self.net.gen_img(torch.zeros(1, 512, 32, 32), style_codes, onehot)

            loss, loss_dict, id_logs = self.calc_loss(target, recon_i, mask)
            
            loss.backward()
            optimizer_net.step()  

        self.net.eval()
        return self.net
        


    def configure_optimizer(self):
        self.params = list(filter(lambda p: p.requires_grad, list(self.net.parameters())))
        print(len(self.params))
        
        if self.opts.opt_name == 'adam':
            optimizer = torch.optim.Adam(self.params, lr=self.opts.finetune_learning_rate)
        else:
            optimizer = Ranger(self.params, lr=self.opts.finetune_learning_rate)
            
        return optimizer


    def invertion(self, sample_idx):
        """
           类似GAN-inversion，得到一张图片的 style code 和 structure code
            
        Args:
            sample_idx (int)
        
        Return:
            num_pairs (int, optional): [description]. Defaults to 20.
            save_dir (str, optional): [description]. Defaults to "".
        """
        
        # 随机取
        img_name = os.path.basename(self.test_ds.imgs[sample_idx]).split(".")[0]
        intermediate_folder = os.path.join(self.opts.output_dir,img_name)
        os.makedirs(intermediate_folder, exist_ok=True)
        
        img, mask, mask_vis = self.test_ds[sample_idx]
        
        img, mask = img.unsqueeze(0), mask.unsqueeze(0)
        img = img.to(self.opts.device).float()
        mask = (mask * 255).long().to(self.opts.device)
        # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
        onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
        
        # 原始的style codes
        with torch.no_grad():
            # struc_code, style_codes = self.net.get_style(img, onehot)
            style_vectors, struc_code = self.net.get_style_vectors(img, onehot)
            style_codes = self.net.cal_style_codes(style_vectors)
            
        # struc_code, style_codes = torch.zeros(1,512,16,16).to(self.opts.device), torch.randn(1,12,18,512).to(self.opts.device)
        # ==========  优化前 重建的图片  ==========  
        # channels = {
        #     4: 512,
        #     8: 512,
        #     16: 512,
        #     32: 512,
        #     64: 256 * 2,
        #     128: 128 * 2,
        #     256: 64 * 2,
        #     512: 32 * 2,
        #     1024: 16 * 2,
        # }
        # noise = [torch.randn(1,512,4,4).to(self.opts.device)]
        # for i in [8,16,32,64,128,256,512,1024]:
        #     noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
        #     noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
            
        recon, _, structure_feats = self.net.gen_img(torch.zeros_like(struc_code), style_codes, onehot)
                                                    #  randomize_noise=False,noise=noise)
        
        img_vis = torch_utils.tensor2im(img[0])
        img_vis.save(os.path.join(self.opts.output_dir, img_name, img_name+"_gt.png"))
        recon_vis = torch_utils.tensor2im(recon[0])
        recon_vis.save(os.path.join(self.opts.output_dir, img_name, img_name+"_recon.png"))
        ## =======================================
        
        optimizer_W, latent = self.setup_W_optimizer(style_vectors,noise_init=None)
        pbar = tqdm(range(self.opts.W_steps), desc='Optimizing style code...', leave=False)
        for step in pbar:
            optimizer_W.zero_grad()
            
            style_codes = self.net.cal_style_codes(latent)
            
            recon_i, _, structure_feats_i  = self.net.gen_img(torch.zeros_like(struc_code), style_codes, onehot)
                                                            #   randomize_noise=False,noise=noise)
            
            # recon_i = recon
            
            loss, loss_dict, id_logs = self.calc_loss(img, recon_i, mask)
            
            loss.backward()
            optimizer_W.step()

            if self.opts.verbose:
                verbose_str = "[%s]"%img_name
                for k,v in loss_dict.items():
                    if k[:4]=="loss":
                        verbose_str += (k + " : {:.4f}, ".format(v))
                pbar.set_description(verbose_str)

            if self.opts.save_intermediate and (step+1) % self.opts.save_interval== 0:
                self.save_W_intermediate_results(img_name, recon_i, latent, step+1)
        
        self.save_W_intermediate_results(img_name, recon_i, latent, self.opts.W_steps, noise=None)

        
    def save_W_intermediate_results(self, img_name, gen_im, latent, step, noise=None):
              
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent.detach().cpu().numpy()
        
        intermediate_folder = os.path.join(self.opts.output_dir,img_name)
        os.makedirs(intermediate_folder, exist_ok=True)

        latent_path = os.path.join(intermediate_folder, f'{img_name}_{step:04}.npy')
        image_path = os.path.join(intermediate_folder, f'{img_name}_{step:04}.png')

        save_im.save(image_path)
        
        if noise is not None:
            save_stats = {}
            save_stats["style_code"] = save_latent
            save_stats["noise"] = [n.detach().cpu().numpy() for n in noise]
            torch.save(save_stats,latent_path)
        else:
            np.save(latent_path, save_latent)


    # @torch.no_grad()
    def swapping_face_optimization(
            self, driven_init_styleVec_path,
            driven, driven_mask,
            target, target_orig_mask, 
        ):
        """
        换脸第二阶段的 optimization。
          
        优化的目标是:
            (1) 使得在 hair, background, belowFace 部分和 target图片一样;
            (2) 同时约束脸部和 driven 的脸部一样
          
        用带 mask 的 loss 来完成
            
        Args:
            driven_init_styleVec_path (str): 第一阶段序列化的 style vectors
            driven (str): 第一阶段换脸的结果
            driven_mask (str):  驱动 + 超分后的图片对应的mask, 再经手动UI编辑后的mask, 已经转成我们用的12个类别
            target (str): target 图片
            target_orig_mask (str): target 图片原始对应的mask, 记得要转到 我们用的12个类别
        """
        
        froom, to = os.path.basename(driven).split('_')[1], os.path.basename(driven).split('_')[3].split('.')[0]
        sample_folder = os.path.join(self.opts.output_dir, "%s_%s"%(froom,to))
        os.makedirs(sample_folder, exist_ok=True)
        
        # 读数据
        target_img = Image.open(target).convert('RGB')
        target_img = self.img_transform(target_img).to(self.opts.device).float().unsqueeze(0)
        target_origin_label = Image.open(target_orig_mask).convert('L')
        target_origin_label = (self.label_transform_w_converter(target_origin_label)* 255).long().to(self.opts.device).unsqueeze(0)
        # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
        target_origin_onehot = torch_utils.labelMap2OneHot(target_origin_label, num_cls=self.opts.num_seg_cls)
        
        driven_img = Image.open(driven).convert('RGB')
        driven_img = self.img_transform(driven_img).to(self.opts.device).float().unsqueeze(0)
        driven_label = Image.open(driven_mask).convert('L')
        driven_label = (self.label_transform_wo_converter(driven_label)* 255).long().to(self.opts.device).unsqueeze(0)
        # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
        driven_onehot = torch_utils.labelMap2OneHot(driven_label, num_cls=self.opts.num_seg_cls)

        
        # 初始化的style vectors
        style_vectors_init =  torch.load(driven_init_styleVec_path).to(self.opts.device)
        
        # style_codes_init = self.net.cal_style_codes(style_vectors_init)
        # with torch.no_grad():    
        #     recon, _, structure_feats = self.net.gen_img(torch.zeros(1,512,32,32).to(self.opts.device), style_codes_init, driven_onehot)
        #                                                 #  randomize_noise=False,noise=noise)
        
        # target_img_vis = torch_utils.tensor2im(target_img[0])
        # target_img_vis.save(os.path.join(self.opts.output_dir, "%s_to_%s_target.png"%(froom ,to)))
        # driven_img_vis = torch_utils.tensor2im(driven_img[0])
        # driven_img_vis.save(os.path.join(self.opts.output_dir, "%s_to_%s_driven.png")%(froom ,to))
        ## =======================================
        
        optimizer_W, latent = self.setup_W_optimizer(style_vectors_init, noise_init=None)
        pbar = tqdm(range(self.opts.W_steps), desc='Optimizing style code...', leave=False)
        for step in pbar:
            optimizer_W.zero_grad()
            
            style_codes = self.net.cal_style_codes(latent)
            
            recon_i, _, structure_feats_i  = self.net.gen_img(torch.zeros(1,512,32,32).to(self.opts.device), style_codes, driven_onehot)
                                                            #   randomize_noise=False,noise=noise)
            
            loss, loss_dict, id_logs = self.calc_loss_swappingFace(target_img, recon_i, driven_img, driven_label)
            
            loss.backward()
            optimizer_W.step()

            if self.opts.verbose:
                verbose_str = "[%s_%s]"%(froom,to)
                for k,v in loss_dict.items():
                    if k[:4]=="loss":
                        verbose_str += (k + " : {:.4f}, ".format(v))
                pbar.set_description(verbose_str)

            if self.opts.save_intermediate and (step+1) % self.opts.save_interval== 0:
                self.save_W_intermediate_results("%s_%s"%(froom,to), recon_i, latent, step+1)
        
        self.save_W_intermediate_results("%s_%s"%(froom, to), recon_i, latent, self.opts.W_steps, noise=None)
            
    # @torch.no_grad()
    def video_swapping_face_optimization(
            self, driven_init_styleVec_path,
            driven, driven_mask,
            target, target_mask, 
            save_folder
        ):
        """
        视频换脸第二阶段的 optimization。
          
        优化的目标是:
            (1) 使得在 hair, background, belowFace 部分和 target图片一样;
            (2) 同时约束脸部和 driven 的脸部一样
          
        用带 mask 的 loss 来完成
            
        Args:
            driven_init_styleVec_path (str): 第一阶段序列化的 style vectors
            driven (str): 第一阶段换脸的结果
            driven_mask (str):  驱动 + 超分后的图片对应的mask, 再经手动UI编辑后的mask, 已经转成我们用的12个类别
            target (str/PIL.image): target 图片
            target_orig_mask (str/PIL.image): target 图片原始对应的mask,  已经转成我们用的12个类别
            save_folder (str): 每张图片保持的文件夹
        """
        # 读数据
        target_img = Image.open(target).convert('RGB') if isinstance(target, str) else target
        target_img = self.img_transform(target_img).to(self.opts.device).float().unsqueeze(0)
        target_label = Image.open(target_mask).convert('L') if isinstance(target_mask, str) else target_mask
        target_label = (self.label_transform_wo_converter(target_label)* 255).long().to(self.opts.device).unsqueeze(0)
        target_onehot = torch_utils.labelMap2OneHot(target_label, num_cls=self.opts.num_seg_cls)
        
        driven_img = Image.open(driven).convert('RGB')
        driven_img = self.img_transform(driven_img).to(self.opts.device).float().unsqueeze(0)
        driven_label = Image.open(driven_mask).convert('L')
        driven_label = (self.label_transform_wo_converter(driven_label)* 255).long().to(self.opts.device).unsqueeze(0)
        driven_onehot = torch_utils.labelMap2OneHot(driven_label, num_cls=self.opts.num_seg_cls)

        
        # 初始化的style vectors
        style_vectors_init =  torch.load(driven_init_styleVec_path).to(self.opts.device)
        
        # style_codes_init = self.net.cal_style_codes(style_vectors_init)
        # with torch.no_grad():    
        #     recon, _, structure_feats = self.net.gen_img(torch.zeros(1,512,32,32).to(self.opts.device), style_codes_init, driven_onehot)
        #                                                 #  randomize_noise=False,noise=noise)
        
        # target_img_vis = torch_utils.tensor2im(target_img[0])
        # target_img_vis.save(os.path.join(self.opts.output_dir, "%s_to_%s_target.png"%(froom ,to)))
        # driven_img_vis = torch_utils.tensor2im(driven_img[0])
        # driven_img_vis.save(os.path.join(self.opts.output_dir, "%s_to_%s_driven.png")%(froom ,to))
        ## =======================================
        
        optimizer_W, latent = self.setup_W_optimizer(style_vectors_init, noise_init=None)
        pbar = tqdm(range(self.opts.W_steps), desc='Optimizing style code...', leave=False)
        for step in pbar:
            optimizer_W.zero_grad()
            
            style_codes = self.net.cal_style_codes(latent)
            
            recon_i, _, structure_feats_i  = self.net.gen_img(torch.zeros(1,512,32,32).to(self.opts.device), style_codes, driven_onehot)
                                                            #   randomize_noise=False,noise=noise)
            
            loss, loss_dict, id_logs = self.calc_loss_swappingFace(target_img, recon_i, driven_img, driven_label)
            
            loss.backward()
            optimizer_W.step()

            if self.opts.verbose:
                verbose_str = ""
                for k,v in loss_dict.items():
                    if k[:4]=="loss":
                        verbose_str += (k + " : {:.4f}, ".format(v))
                pbar.set_description(verbose_str)

        self.save_W_intermediate_results(save_folder, recon_i, latent, self.opts.W_steps, noise=None)
        
        return torch_utils.tensor2im(recon_i[0])
        
if __name__ == '__main__':
    opts = OptimOptions().parse()
    print("make dir %s"%opts.output_dir)
    os.makedirs(opts.output_dir, exist_ok=True)
    
    optimizer = Optimizer(opts)
    # for i in range(opts.begin_idx,opts.begin_idx+100):
    
    """
    high_quality_imgs = [
        21,22,25,35,41,51,60,61,62,63,69,83,90,97,109,
        130,132,139,156,161,163,171,176,179,183,209,248,
        256,258,260,261,272,286,287,297,299,301,320,328,
        337,354,356,359,362,363,370,374,378,389,392,394,
        396,398,409,414,421,426,479,494,497,500,527,534,
        537,542,547,551,556,558,605,613,618,621,622,643,
        669,671,686,688,689,701,707,723,737,754,761,783,
        793,800,831,833,835,837,840,861,872,898,907,935,
        965,978,1005,1010,1011,1026,1030,1032,1039,1042,
        1095,1141,1142,1155,1206,1220,1222,1258,1266,1275,
        1284,1308,1337,1353,1357,1336,1376,1383,1426,1438,
        1441,1442,1490,1507,1514,1524,1575,1581,1607,1698,
        1721,1756,1766,1802,1833,1927,1980,1989
    ]

    img_idx = [28031,28352,28381,28006,28314,28298,28363,28394,28101,28022,28380,28221,
               29258,28541,29057,28092,28905,29002, 28408, 29438]
    
    recon_img_idx = [28299,28494,29258,29766,29357,28542,28022,29337]
    # img_idx = [29002, 28408, 29438]
    for i in [28479,29275]:
    # for i in recon_img_idx:
    # for i in range(29500,30000):
        optimizer.invertion(i-28000)

    img_path = "/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/tmp/D_liudehua_to_2.png"
    mask_path = "/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/tmp/D_mask.png"

    optimizer.inversion_img(img_path=img_path, mask_path=mask_path)

    optimizer.swapping_face_optimization(driven_init_styleVec_path="/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/tmp/swapped_style_vec.pt",
                                            driven="/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/tmp/D_liudehua_to_2.png",
                                            driven_mask="/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/tmp/swappedMask.png",
                                            target="/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/tmp/T_cropped.png",
                                            target_orig_mask="/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/tmp/T_mask.png")
    """

    target_path = "/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/vis/D_liudehua_to_2.png"
    mask_path = "/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/vis/D_mask.png"
    style_vectors_path = "/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/optim_res/D_liudehua_to_2/D_liudehua_to_2_0200.npy"
    optimizer.finetune_net(target_path=target_path, mask_path=mask_path, style_vectors_path=style_vectors_path)
