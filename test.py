"""
This file runs the main training/val loop
"""
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.dataset import CelebAHQDataset, get_transforms, TO_TENSOR, NORMALIZE, MASK_CONVERT_TF, MASK_CONVERT_TF_DETAILED
from models.networks import Net, Net2, Net3, MultiScaleNet
from models.encoder_with_optim import EncoderPlusOptimNet
from options.test_options import TestOptions
import os
import json
import sys
import pprint
import torch
from utils import torch_utils
from tqdm import tqdm
import numpy as np
from PIL import Image

sys.path.append(".")
sys.path.append("..")


class Tester:
    def __init__(self,opts):
        self.opts = opts
        self.test_ds = CelebAHQDataset(dataset_root=self.opts.dataset_root, mode="test",
                                       img_transform=transforms.Compose(
                                           [TO_TENSOR, NORMALIZE]),
                                       label_transform=transforms.Compose(
                                           [ MASK_CONVERT_TF_DETAILED,TO_TENSOR]), # MASK_CONVERT_TF,
                                       fraction=self.opts.ds_frac)
        print(f"Number of test samples: {len(self.test_ds)}")

        self.test_dataloader = DataLoader(self.test_ds, batch_size=self.opts.test_batch_size,
                                          shuffle=False, num_workers=int(self.opts.test_workers), drop_last=False)
        
        
        assert self.opts.checkpoint_path is not None, "please specify the pre-trained weights!"
        self.net = Net3(self.opts).eval().to(self.opts.device)
        
        ckpt_dict=torch.load(self.opts.checkpoint_path)
        self.net.latent_avg = ckpt_dict['latent_avg'].to(self.opts.device) if self.opts.start_from_latent_avg else None
        self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
        print("Load pre-trained weights.")        

    # @torch.no_grad()
    # def test(self,num_imgs=20, save_dir=""):
    #     cnt = 0
        
    #     for batch_idx, batch in enumerate(self.test_dataloader):
    #         (img1, img2), (mask1, mask2), (mask1_vis, mask1_vis) = batch
    #         img1, img2 = img1.to(self.opts.device).float(), img2.to(
    #             self.opts.device).float()
    #         mask1 = (mask1*255).long().to(self.opts.device)
    #         mask2 = (mask2*255).long().to(self.opts.device)

    #         # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
    #         onehot1 = torch_utils.labelMap2OneHot(
    #             mask1, num_cls=self.opts.num_seg_cls)
    #         onehot2 = torch_utils.labelMap2OneHot(
    #             mask2, num_cls=self.opts.num_seg_cls)

    #         swap = False
    #         recon1, latent1 = self.net(img1, onehot1, return_latents=True)
    #         recon2, latent2 = self.net(img2, onehot2, return_latents=True)

    #         imgs_1 = self.parse_images(onehot1, img1, recon1)
    #         imgs_2 = self.parse_images(onehot2, img2, recon2)

    #         self.save_imgs(imgs_1, save_name="%04d" % (cnt))
    #         self.save_imgs(imgs_2, save_name="%04d" % (cnt+1))
    #         cnt += len(img1)*2
            
    #         if cnt>num_imgs:
    #             break
    
    # 补偿网络测试
    @torch.no_grad()
    def compensate_test(self,num_imgs=20, save_dir=""):
        cnt = 0
        for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):

            img, recon_img, optimed_style_code, mask, mask_vis = batch

            sample_name = os.path.basename(self.test_ds.imgs[batch_idx])[:-4]
            
            img = img.to(self.opts.device).float()
            recon_img = recon_img.to(self.opts.device).float()
            optimed_style_code = optimed_style_code.to(self.opts.device).float()
            mask = (mask*255).long().to(self.opts.device)
            # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
            onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
            
            diffMap = img - recon_img
            
            res_feats, base_codes, compensated_codes = self.net.get_style(img, onehot, diff_img=diffMap)    
            
            # 1. 不用补偿style code 的重建结果
            recon_1 = self.net.gen_img(res_feats, base_codes, onehot)
            # 2. 用了补偿style code 的重建结果
            recon_2 = self.net.gen_img(res_feats, base_codes + compensated_codes, onehot)
            
            
            imgs = {
                'input_face': torch_utils.tensor2im(img[0]),
                'input_mask': torch_utils.tensor2map(onehot[0]),
                'recon_face_wo_compensation': torch_utils.tensor2im(recon_1[0]),
                'recon_face_w_compensation': torch_utils.tensor2im(recon_2[0]),
            }
            self.save_imgs(imgs, save_name=sample_name)
            
            cnt += len(img)
            
            if cnt>num_imgs:
                break
            
    @torch.no_grad()
    def test(self,num_imgs=20, save_dir="",begin_idx=0):
        cnt = 0
        for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):

            if batch_idx < begin_idx:
                continue
            
            img, mask, mask_vis = batch

            sample_name = os.path.basename(self.test_ds.imgs[batch_idx])[:-4]
            
            img = img.to(self.opts.device).float()
            mask = (mask*255).long().to(self.opts.device)
            # [bs,1,H,W]的mask 转成one-hot格式，即[bs,#seg_cls,H,W]
            onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)
                
            recon, structure_codes_GT, latent = self.net(img, onehot, return_latents=True)
            
            imgs = self.parse_images(onehot, img, recon)
            self.save_imgs(imgs, save_name=sample_name)
            
            cnt += len(img)
            
            if cnt>num_imgs:
                break
            
    def save_imgs(self, img_data, save_name):
        # # gt 和 recon 放在一张图·
        # arr = np.hstack((
        #     np.array(img_data["input_face"]),
        #     np.array(img_data["recon_face"])
        # ))
        # Image.fromarray(arr).save(os.path.join(self.opts.save_dir, "%s.png" % save_name))
        
        for k, v in img_data.items():
            if k=="recon_face":
                v.save(os.path.join(self.opts.save_dir, "%s_%s.png" % (save_name, k)))

    
    def parse_images(self, mask, img, recon):
        cur_im_data = {
            'input_face': torch_utils.tensor2im(img[0]),
            'input_mask': torch_utils.tensor2map(mask[0]),
            'recon_face': torch_utils.tensor2im(recon[0]),
        }
        return cur_im_data


if __name__ == '__main__':
    opts = TestOptions().parse()
    os.makedirs(opts.save_dir, exist_ok=True)
    tester = Tester(opts)
    tester.test(num_imgs=2000,begin_idx=0)
