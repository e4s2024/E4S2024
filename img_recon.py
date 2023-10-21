"""
This file runs the main training/val loop
"""
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.dataset import CelebAHQDataset, get_transforms, TO_TENSOR, NORMALIZE, MASK_CONVERT_TF, MASK_CONVERT_TF_DETAILED
from models.networks import Net3
from options.test_options import TestOptions
import glob
import os
import json
import sys
import pprint
import torch
from utils import torch_utils
from tqdm import tqdm
import numpy as np
from PIL import Image
from options.swap_face_options import SwapFaceOptions
from swap_face_fine.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps

sys.path.append(".")
sys.path.append("..")


# 重建一张/几张图片
@torch.no_grad()
def recon_imgs(opts, imgs_path, out_dir="./tmp"): 
    net = Net3(opts).eval().to(opts.device)
    ckpt_dict=torch.load("/apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/ablation_study/v_15_baseline_seg12_finetuneGD_8A100_remainLyrIdx13_flip_FFHQ_300KIters/checkpoints/iteration_300000.pt")
    net.latent_avg = ckpt_dict['latent_avg'].to(opts.device)
    net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
    print("Load pre-trained weights.")        

    # face parsing 模型
    faceParsing_ckpt = "./pretrained/faceseg/79999_iter.pth"
    faceParsing_model = init_faceParsing_pretrained_model(faceParsing_ckpt)

    for idx, img_path in enumerate(tqdm(imgs_path)):

        img_pil = Image.open(img_path).convert("RGB")
        sample_name = os.path.basename(img_path)[:-4]
        mask = faceParsing_demo(faceParsing_model, img_pil, convert_to_seg12=True)
        
        # wrap data 
        img = transforms.Compose([TO_TENSOR, NORMALIZE])(img_pil)
        img = img.to(opts.device).float().unsqueeze(0)
        
        mask = transforms.Compose([TO_TENSOR])(Image.fromarray(mask))
        mask = (mask*255).long().to(opts.device).unsqueeze(0)
        onehot = torch_utils.labelMap2OneHot(mask, num_cls = opts.num_seg_cls)
        
        recon, structure_codes_GT, latent = net(img, onehot, return_latents=True)
        
        imgs = parse_images(onehot, img, recon)
        
        # mask, gt 和 recon 放在一张图·
        arr = np.hstack((
            np.array(imgs["input_mask"].resize((1024,1024))),
            np.array(imgs["input_face"]),
            np.array(imgs["recon_face"])
        ))
        Image.fromarray(arr).save(os.path.join(out_dir, "%s_recon.png" % sample_name))
        

def parse_images(mask, img, recon):
    cur_im_data = {
        'input_face': torch_utils.tensor2im(img[0]),
        'input_mask': torch_utils.tensor2map(mask[0]),
        'recon_face': torch_utils.tensor2im(recon[0]),
    }
    return cur_im_data


# 重建一张 + 修改后的mask编辑
@torch.no_grad()
def recon_then_edit(opts, samples, out_dir="./tmp2"): 
    
    # 固定noise
    channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * 2,
            128: 128 * 2,
            256: 64 * 2,
            512: 32 * 2,
            1024: 16 * 2,
        }
    noise = [torch.randn(1,512,4,4).to(opts.device)]
    for i in [8,16,32,64,128,256,512,1024]:
        noise.append(torch.randn(1,channels[i],i,i).to(opts.device))
        noise.append(torch.randn(1,channels[i],i,i).to(opts.device))
        
    net = Net3(opts).eval().to(opts.device)
    ckpt_dict=torch.load("/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/v_15_hybrid_stage1_seg12_finetuneGD_8A100_pspHyperParas_remainLyrIdx13_flip_200KIters/checkpoints/iteration_120000.pt")
    net.latent_avg = ckpt_dict['latent_avg'].to(opts.device)
    net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
    print("Load pre-trained weights.")        

    
    for sample_name in sampels:
        img_path = os.path.join("/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/images/%s.jpg"%sample_name)
        orig_mask_path = os.path.join("/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/labels/%s.png"%sample_name)
        edit_mask_path = os.path.join("./tmp2/%s_edit_mask.png"%sample_name)
        
        img_pil = Image.open(img_path).convert("RGB")
        sample_name = os.path.basename(img_path)[:-4]
        
        mask = Image.open(orig_mask_path).convert("L")
        edit_mask = Image.open(edit_mask_path).convert("L")
        
        # wrap data
        img = transforms.Compose([TO_TENSOR, NORMALIZE])(img_pil)
        img = img.to(opts.device).float().unsqueeze(0)
        
        mask = transforms.Compose([MASK_CONVERT_TF_DETAILED, TO_TENSOR])(mask)
        mask = (mask*255).long().to(opts.device).unsqueeze(0)
        onehot = torch_utils.labelMap2OneHot(mask, num_cls = opts.num_seg_cls)
        
        edit_mask = transforms.Compose([MASK_CONVERT_TF_DETAILED, TO_TENSOR])(edit_mask)
        edit_mask = (edit_mask*255).long().to(opts.device).unsqueeze(0)
        edit_onehot = torch_utils.labelMap2OneHot(edit_mask, num_cls = opts.num_seg_cls)
        
        style_vecs, _ = net.get_style_vectors(img, onehot)
        style_codes = net.cal_style_codes(style_vecs)
        
        
        recon, _, _ = net.gen_img(torch.zeros(1,512,32,32).to(onehot.device), 
                                        style_codes, onehot,
                                        randomize_noise=False,noise=noise)

        
        edit_result, _, _ = net.gen_img(torch.zeros(1,512,32,32).to(onehot.device), 
                                        style_codes, edit_onehot,
                                        randomize_noise=False,noise=noise)
        
        recon_pil = torch_utils.tensor2im(recon[0])
        recon_pil.save(os.path.join(out_dir, "%s_recon.png" % sample_name))
        
        edit_pil = torch_utils.tensor2im(edit_result[0])
        edit_pil.save(os.path.join(out_dir, "%s_edit.png" % sample_name))
        
        orig_mask_pil = torch_utils.tensor2map(onehot[0])
        orig_mask_pil.save(os.path.join(out_dir, "%s_orig_maskVis.png" % sample_name))
        
        edit_mask_pil = torch_utils.tensor2map(edit_onehot[0])
        edit_mask_pil.save(os.path.join(out_dir, "%s_edit_maskVis.png" % sample_name))

        
        

if __name__ == '__main__':
    opts = SwapFaceOptions().parse()
    # img_paths = glob.glob(os.path.join("/apdcephfs/share_1290939/zhianliu/datasets/video_swapping_demo/celebrate/crop/*"))
    # recon_imgs(opts, img_paths)
    
    
    sampels = ["28114","28584","28733","29354"]
    recon_then_edit(opts, sampels)