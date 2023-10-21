import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.dataset import CelebAHQDataset, get_transforms, TO_TENSOR, NORMALIZE, MASK_CONVERT_TF, MASK_CONVERT_TF_DETAILED
from models.networks import Net,Net2,Net3,NetStage2
from options.edit_options import EditOptions
import os
import json
import sys
import pprint
import torch
from utils import torch_utils
import random
import torch.nn.functional as F
from torchvision.utils import make_grid
import shutil
from PIL import Image
from utils.morphology import dilation
import copy
import cv2

sys.path.append(".")
sys.path.append("..")

label_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/labels"
image_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/images"
lael_vis_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/vis"

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
# 12个属性
faceParser_label_list_detailed = ['background', 'lip', 'eyebrows',
                         'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface','mouth','eye_glass','ear_rings']



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

def smooth_face_boundry(image, dst_image, mask, radius=0, sigma=0.0):
    # 把 image 贴到 dst_image 上去, 其中mask是 image内脸对应的mask
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask) # mask 需要是 [0,255] 范围
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    pasted_image.alpha_composite(image_masked)
    return pasted_image

def bg_pasting(img, mask, img_orig):
    # Gaussian blending with mask
    # 处理一下mask， 得到face 区域和 dialation区域
    outer_dilation = 2
    mask_bg = logical_or_reduce(*[mask == clz for clz in [0,11]])  # 如果是视频换脸，考虑把头发也弄进来当做背景的一部分
    is_foreground = torch.logical_not(mask_bg)
    foreground_mask = is_foreground.float()
    
    content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation = outer_dilation)
    
    content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
    content_mask_image = Image.fromarray(255*content_mask[0,0,:,:].cpu().numpy().astype(np.uint8))
    border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
    full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=False)
    full_mask_image = Image.fromarray(255*full_mask[0,0,:,:].cpu().numpy().astype(np.uint8))
    
    # 直接贴，但是脸部和背景融合的地方 smooth一下
    if outer_dilation == 0:
        pasted_image = smooth_face_boundry(img, img_orig, content_mask_image, radius=outer_dilation)
    else:
        pasted_image = smooth_face_boundry(img, img_orig, full_mask_image, radius=outer_dilation)
    
    return pasted_image

class Editor:
    def __init__(self,opts):
        self.opts = opts
        # assert self.opts.swap_comp in faceParser_label_list, "The swap comp. %s is not supported."%self.opts.swap_comp
        
        self.test_ds = CelebAHQDataset(dataset_root=self.opts.dataset_root, mode="test",
                                       img_transform=transforms.Compose(
                                           [TO_TENSOR, NORMALIZE]),
                                       label_transform=transforms.Compose(
                                           [ MASK_CONVERT_TF_DETAILED,TO_TENSOR]), # MASK_CONVERT_TF,
                                       fraction=self.opts.ds_frac)
        print(f"Number of test samples: {len(self.test_ds)}")

        
        assert self.opts.checkpoint_path is not None, "please specify the pre-trained weights!"
        self.net = Net3(self.opts).eval().to(self.opts.device)
        
        ckpt_dict=torch.load(self.opts.checkpoint_path)
        self.net.latent_avg = ckpt_dict['latent_avg'].to(self.opts.device) if self.opts.start_from_latent_avg else None
        if self.opts.load_ema:
            self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict_ema"],prefix="module."))
        else:            
            self.net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
            
        print("Load pre-trained weights.")    
          
 
    def swap_comp_style(self, style_codes1, style_codes2, style_layer_idx=0,swap_comp=None):
        """交换某个component的 style code

        Args:
            style_codes1 (Tensor): with shape [1,#comp,18,512]
            style_codes2 (Tensor): with shape [1,#comp,18,512]
        """
        if swap_comp is None:
            comp_indices = [faceParser_label_list_detailed.index(comp) for comp in self.opts.swap_comp.split(",")]
        else:
            comp_indices = [faceParser_label_list_detailed.index(comp) for comp in swap_comp.split(",")]
        
        style_codes1_cp, style_codes2_cp  = style_codes1.clone(), style_codes2.clone()
        
        for comp_idx in comp_indices:
            style_codes1_cp[:,comp_idx,style_layer_idx:,:] =  style_codes2[:,comp_idx,style_layer_idx:,:]
            style_codes2_cp[:,comp_idx,style_layer_idx:,:] =  style_codes1[:,comp_idx,style_layer_idx:,:]
        
        # style_codes1_cp[:,:,style_layer_idx:,:] =  style_codes2[:,:,style_layer_idx:,:]
        # style_codes2_cp[:,:,style_layer_idx:,:] =  style_codes1[:,:,style_layer_idx:,:]
            
        return style_codes1_cp, style_codes2_cp

    def swap_comp_style_vector(self, style_vectors1, style_vectors2, swap_comp=None):
        """交换某个component的 style vectors

        Args:
            style_vectors1 (Tensor): with shape [1,#comp,512]
            style_vectors1 (Tensor): with shape [1,#comp,512]
        """
        if swap_comp is None:
            comp_indices = [faceParser_label_list_detailed.index(comp) for comp in self.opts.swap_comp.split(",")]
        else:
            comp_indices = [faceParser_label_list_detailed.index(comp) for comp in swap_comp.split(",")]
        
        style_vectors1_cp, style_vectors2_cp  = style_vectors1.clone(), style_vectors2.clone()
        
        for comp_idx in comp_indices:
            style_vectors1_cp[:,comp_idx,:] =  style_vectors2[:,comp_idx,:]
            style_vectors2_cp[:,comp_idx,:] =  style_vectors1[:,comp_idx,:]
        
        return style_vectors1_cp, style_vectors2_cp
    
        
    def swap_comp_mask(self, mask1, mask2, swap_comp):
        """交换某个component的 shape

        Args:
            mask1 (Tensor): with shape [1,1,H,W]
            mask2 (Tensor): with shape [1,1,H,W]
        """
        comp_indices = [faceParser_label_list_detailed.index(comp) for comp in swap_comp.split(",")]
        
        mask1_cp, mask2_cp  = mask1.clone(), mask2.clone()
        
    
        # 先清除掉指定的区域    
        for comp_idx in comp_indices:
            # 原始图片对应的local区域
            mask1_local_comp_region = (mask1==comp_idx)
            mask2_local_comp_region = (mask2==comp_idx)
            
            # 先置为皮肤
            mask1_cp[mask1_local_comp_region]=6 
            mask2_cp[mask2_local_comp_region]=6
        
        # onehot1_cp_i = torch_utils.labelMap2OneHot(mask1_cp, num_cls=self.opts.num_seg_cls)
        # torch_utils.tensor2map(onehot1_cp_i[0],shown_mask_indices=None).save("./tmp/img1_mask.png")
        
        # onehot2_cp_i = torch_utils.labelMap2OneHot(mask2_cp, num_cls=self.opts.num_seg_cls)
        # torch_utils.tensor2map(onehot2_cp_i[0],shown_mask_indices=None).save("./tmp/img2_mask.png")
        
        
        # 再置为target mask 
        for comp_idx in comp_indices:
            # 原始图片对应的local区域
            mask1_local_comp_region = (mask1==comp_idx)
            mask2_local_comp_region = (mask2==comp_idx)
            
            mask1_cp[mask2_local_comp_region]=comp_idx
            mask2_cp[mask1_local_comp_region]=comp_idx
            
            # onehot1_cp_i = torch_utils.labelMap2OneHot(mask1_cp, num_cls=self.opts.num_seg_cls)
            # torch_utils.tensor2map(onehot1_cp_i[0],shown_mask_indices=None).save("./tmp/img1_mask_%d.png"%comp_idx)
            
            # onehot2_cp_i = torch_utils.labelMap2OneHot(mask2_cp, num_cls=self.opts.num_seg_cls)
            # torch_utils.tensor2map(onehot2_cp_i[0],shown_mask_indices=None).save("./tmp/img2_mask_%d.png"%comp_idx)
            
        onehot1_cp = torch_utils.labelMap2OneHot(mask1_cp, num_cls=self.opts.num_seg_cls)
        onehot2_cp = torch_utils.labelMap2OneHot(mask2_cp, num_cls=self.opts.num_seg_cls)
            
        return onehot1_cp, onehot2_cp
    
    def mask_translation(self, mask, comp_name='mouth',dy=-200):
        """将某个component的 mask 的进行平移

        Args:
            mask1 (Tensor): with shape [1,1,H,W]
        """
        comp_idx = faceParser_label_list.index(comp_name)
        mask_cp = mask.clone()
        
        comp_region = (mask==comp_idx)
        mask_cp[comp_region] = 6  # 先置为皮肤
        
        for i in range(comp_region.size(2)):
            for j in range(comp_region.size(3)):
                if comp_region[0,0,i,j]:
                    mask_cp[0,0,i+dy,j] = comp_idx 

        onehot = torch_utils.labelMap2OneHot(mask_cp, num_cls=self.opts.num_seg_cls)
        
        # torch_utils.tensor2map(onehot[0]).save("./tmp/a.png")
        return onehot
        
    def prepare_img_pairs(self, edit_mode):
         # 指定一些编辑的图片
        if edit_mode=="comp_style": # 交换某个属性（例如,头发）后的style codes
            self.img1_names = ['28363', '28340', '28314', '28101', '28380', '28184',
                                '28097', '28398', '28303', '28053', '28159', '28191',
                                '28282', '28031', '28203', '28381', '28354', '28210', 
                                '28020', '28143', '28034']
            self.img2_names = ['28394', '28026', '28298', '28022', '28221', '28088', 
                                '28395', '28172', '28253', '28073', '28286', '28291', 
                                '28124', '28352', '28116', '28006', '28067', '28284', 
                                '28275', '28291', '28174']
        elif edit_mode=="comp_mask": # 交换某个属性（例如,鼻子）后的shape TODO: 改成交互式方式编辑
            # 交换某个component 的shape, 即mask
            self.img1_names = ['28095', '28267', '28106', '28005', '28128',
                                '28365', '28286', '28382', '28187', '28208', 
                                '28015', '28134', '28330', '28164', '28230', 
                                '28280', '28357', '28362', '28194', '28104',
                                ]
            self.img2_names = ['28187', '28158', '28361', '28336', '28087',
                                '28233', '28077', '28061', '28095', '28379', 
                                '28380', '28131', '28091', '28322', '28036', 
                                '28196', '28232', '28056', '28044', '28097',
                                ]
        elif edit_mode=="global_style": # 交换整体的style codes
            self.img1_names = ["28119","28059","28056","28155","28009","28151",
                                 "28162","28163","28189","28191","28238","28248","28304","28350"]
            self.img2_names = ["28058","28021","28192","28374","28162","28290",
                                "28009","28363","28237","28364","28242","28396","28276","28265"]
    
    
    @torch.no_grad()
    def edit(self, num_pairs=20, edit_mode="comp_style"):
        """
            随机选择一对图片,交换某个部位,得到一张新的图片。
            
            有以下几种方式:
            (a) comp_style: 换某个局部的style,例如,img1是黄头发,img2是黑头发,交换二者的style codes,但是各自的mask保持不变
            (b) comp_shape: 换某个局部的mask,例如,img1是光头,img2是长头发,交换二者的mask,但是各自的style code保持不变
            (c) global_style: 换两张图片整体的style,但是各自的mask保持不变
            (d) global_shape: 换两张图片整体的shape,但是各自的style code保持不变。实际上和(c)等效
            
        Args:
            num_pairs (int, optional): [description]. Defaults to 20.
            save_dir (str, optional): [description]. Defaults to "".
        """
        
        
        cnt = 0
        while True:
            # 随机取
            # rand_idx1,rand_idx2 = random.sample(high_quality_imgs, 2)
            # rand_idx1, rand_idx2 = 1442, random.sample(high_quality_imgs, 1)[0]
            rand_idx1,rand_idx2 = 171, 396
            img1_name = os.path.basename(self.test_ds.imgs[rand_idx1]).split(".")[0]
            img2_name = os.path.basename(self.test_ds.imgs[rand_idx2]).split(".")[0]
            img1, mask1, mask_vis1 = self.test_ds[rand_idx1]
            img2, mask2, mask_vis2 = self.test_ds[rand_idx2]
            
            img1, mask1 = img1.unsqueeze(0), mask1.unsqueeze(0)
            img1 = img1.to(self.opts.device).float()
            mask1 = (mask1*255).long().to(self.opts.device)
            # [bs,1,H,W]的mask 转成one-hot格式,即[bs,#seg_cls,H,W]
            onehot1 = torch_utils.labelMap2OneHot(mask1, num_cls=self.opts.num_seg_cls)
            
            img2, mask2 = img2.unsqueeze(0), mask2.unsqueeze(0)
            img2 = img2.to(self.opts.device).float()
            mask2 = (mask2*255).long().to(self.opts.device)
            # [bs,1,H,W]的mask 转成one-hot格式,即[bs,#seg_cls,H,W]
            onehot2 = torch_utils.labelMap2OneHot(mask2, num_cls=self.opts.num_seg_cls)
            
            # 原始的style codes
            style_vectors1, struc_code1 = self.net.get_style_vectors(img1, onehot1)
            style_codes1 = self.net.cal_style_codes(style_vectors1)
            style_vectors2, struc_code2 = self.net.get_style_vectors(img2, onehot2)
            style_codes2 = self.net.cal_style_codes(style_vectors2)
            
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
            noise = [torch.randn(1,512,4,4).to(self.opts.device)]
            for i in [8,16,32,64,128,256,512,1024]:
                noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
                noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
            
            ## ==========  重建后的图片  ==========  
            recon1, _, structure_feats_1 = self.net.gen_img(struc_code1, style_codes1, onehot1,
                                                            randomize_noise=False, noise=noise)
            recon2, _, structure_feats_2 = self.net.gen_img(struc_code2, style_codes2, onehot2,
                                                            randomize_noise=False, noise=noise)
            ## =======================================
            
           
            
            if edit_mode=="comp_style": # 交换某个属性（例如,头发）后的style codes
                # 交换 style vectors
                style_vectors_edited1, style_vectors_edited2 = self.swap_comp_style_vector(style_vectors1, style_vectors2)
                style_codes_edited1 = self.net.cal_style_codes(style_vectors_edited1)
                style_codes_edited2 = self.net.cal_style_codes(style_vectors_edited2)
                struc_codes_edited1, struc_codes_edited2 = struc_code1, struc_code2  # torch.zeros_like(struc_code1), torch.zeros_like(struc_code2) 
                onehot_edited1, onehot_edited2 = onehot1, onehot2
            elif edit_mode=="comp_mask": # 交换某个属性（例如,鼻子）后的shape TODO: 改成交互式方式编辑
                style_codes_edited1, style_codes_edited2 = style_codes1, style_codes2
                struc_codes_edited1, struc_codes_edited2 = struc_code1, struc_code2
                onehot_edited1, onehot_edited2 = self.swap_comp_mask(mask1, mask2,swap_comp=self.opts.swap_comp)
                
                # onehot_edited1 = self.mask_translation(mask1,comp_name="mouth",dy=-200)
                # onehot_edited2 = self.mask_translation(mask2,comp_name="mouth",dy=50)
                
            elif edit_mode=="swap_face": # 换脸
                style_vectors_edited1, style_vectors_edited2 = self.swap_comp_style_vector(style_vectors1, style_vectors2,
                                                                                           swap_comp="skin,nose,belowface,eyes,eyebrows,lip,mouth")
                style_codes_edited1 = self.net.cal_style_codes(style_vectors_edited1)
                style_codes_edited2 = self.net.cal_style_codes(style_vectors_edited2)
                struc_codes_edited1, struc_codes_edited2 = struc_code1, struc_code2
                onehot_edited1, onehot_edited2 = self.swap_comp_mask(mask1, mask2,swap_comp="eyes,nose,lip,mouth")
                
                # onehot_edited1 = self.mask_translation(mask1,comp_name="mouth",dy=-200)
                # onehot_edited2 = self.mask_translation(mask2,comp_name="mouth",dy=50)
    
            elif edit_mode=="global_style": # 交换整体的style codes
                style_codes_edited1, style_codes_edited2 =  style_codes2, style_codes1
                struc_codes_edited1, struc_codes_edited2 = struc_code1, struc_code2 # structure code 不变
                onehot_edited1, onehot_edited2 = onehot1, onehot2
            elif edit_mode=="global_shape": # 交换整体的shape
                style_codes_edited1, style_codes_edited2 = style_codes1, style_codes2
                struc_codes_edited1, struc_codes_edited2 = struc_code1, struc_code2
                onehot_edited1, onehot_edited2 =  onehot2, onehot1 
            else:
                raise NotImplementedError("其他的编辑模式暂时没实现!")
        
            ## ==========  生成编辑后的图片  ==========  
            edit1, _, edit_structure_feats_1 = self.net.gen_img(struc_codes_edited1, style_codes_edited1, onehot_edited1,
                                                                randomize_noise=False,noise=noise)
            edit2, _, edit_structure_feats_2 = self.net.gen_img(struc_codes_edited2, style_codes_edited2, onehot_edited2,
                                                                randomize_noise=False,noise=noise)
            ## =======================================            
            # 保存结果到图片
            vis_dict = {}
            
            # vis_dict["img1"] = torch_utils.tensor2im(img1[0])
            # vis_dict["img2"] = torch_utils.tensor2im(img2[0])
            
            # vis_dict["mask1"] = torch_utils.tensor2map(onehot1[0])
            # vis_dict["mask2"] = torch_utils.tensor2map(onehot2[0])
            
            if edit_mode=="comp_mask" or edit_mode=="swap_face":
                vis_dict["edited_mask1"] = torch_utils.tensor2map(onehot_edited1[0],shown_mask_indices=None)
                vis_dict["edited_mask2"] = torch_utils.tensor2map(onehot_edited2[0],shown_mask_indices=None)
            
            # vis_dict["recon1"] = torch_utils.tensor2im(recon1[0])
            # vis_dict["recon2"] = torch_utils.tensor2im(recon2[0])
            
            # 是否贴回背景
            paste_back = True 
            if paste_back:
                vis_dict["edit1"] = bg_pasting(torch_utils.tensor2im(edit1[0]), mask1, torch_utils.tensor2im(img1[0]))
                vis_dict["edit2"] = bg_pasting(torch_utils.tensor2im(edit2[0]), mask2, torch_utils.tensor2im(img2[0]))
                
            else:
                vis_dict["edit1"] = torch_utils.tensor2im(edit1[0])
                vis_dict["edit2"] = torch_utils.tensor2im(edit2[0])
                   
            self.save_imgs(vis_dict, save_name="%s_%s" % (img1_name,img2_name))
            
            cnt += 1
            
            if cnt== num_pairs:
                break
            
    def save_imgs(self, img_data, save_name):
        for k, v in img_data.items():
            v.save(os.path.join(self.opts.save_dir, "%s_%s.png" % (save_name, k)))
    
    
    def dummy(self, idx):
        img_name = os.path.basename(self.test_ds.imgs[idx]).split(".")[0]
        img, mask, mask_vis = self.test_ds[idx]
        
        img, mask = img.unsqueeze(0), mask.unsqueeze(0)
        img = img.to(self.opts.device).float()
        mask = (mask*255).long().to(self.opts.device)
        # [bs,1,H,W]的mask 转成one-hot格式,即[bs,#seg_cls,H,W]
        onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.opts.num_seg_cls)

        torch_utils.tensor2map(onehot[0]).save("tmp/%d.png"%idx)
    
    @torch.no_grad()
    def interpolation(self,num_pairs=10):
        cnt = 0
        while True:
        # 随机取
            rand_idx1,rand_idx2 = random.sample(high_quality_imgs, 2)
            # rand_idx1,rand_idx2 = 701,833
            img1_name = os.path.basename(self.test_ds.imgs[rand_idx1]).split(".")[0]
            img2_name = os.path.basename(self.test_ds.imgs[rand_idx2]).split(".")[0]
            img1, mask1, mask_vis1 = self.test_ds[rand_idx1]
            img2, mask2, mask_vis2 = self.test_ds[rand_idx2]
            
            img1, mask1 = img1.unsqueeze(0), mask1.unsqueeze(0)
            img1 = img1.to(self.opts.device).float()
            mask1 = (mask1*255).long().to(self.opts.device)
            # [bs,1,H,W]的mask 转成one-hot格式,即[bs,#seg_cls,H,W]
            onehot1 = torch_utils.labelMap2OneHot(mask1, num_cls=self.opts.num_seg_cls)
            
            img2, mask2 = img2.unsqueeze(0), mask2.unsqueeze(0)
            img2 = img2.to(self.opts.device).float()
            mask2 = (mask2*255).long().to(self.opts.device)
            # [bs,1,H,W]的mask 转成one-hot格式,即[bs,#seg_cls,H,W]
            onehot2 = torch_utils.labelMap2OneHot(mask2, num_cls=self.opts.num_seg_cls)
            
            # 原始的style codes
            style_vectors1, struc_code1 = self.net.get_style_vectors(img1, onehot1)
            style_codes1 = self.net.cal_style_codes(style_vectors1)
            style_vectors2, struc_code2 = self.net.get_style_vectors(img2, onehot2)
            style_codes2 = self.net.cal_style_codes(style_vectors2)
            
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
            noise = [torch.randn(1,512,4,4).to(self.opts.device)]
            for i in [8,16,32,64,128,256,512,1024]:
                noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
                noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
            
            ## ==========  重建后的图片  ==========  
            recon1, _, structure_feats_1 = self.net.gen_img(struc_code1, style_codes1, onehot1,
                                                            randomize_noise=False, noise=noise)
            recon2, _, structure_feats_2 = self.net.gen_img(struc_code2, style_codes2, onehot2,
                                                            randomize_noise=False, noise=noise)
       
            recon1_vis = torch_utils.tensor2im(recon1[0])
            recon1_vis.save(os.path.join(self.opts.save_dir,"%s_%s_recon1.png"%(img1_name,img2_name)))
            
            recon2_vis = torch_utils.tensor2im(recon2[0])
            recon2_vis.save(os.path.join(self.opts.save_dir,"%s_%s_recon2.png"%(img1_name,img2_name)))
            
            # ========== 开始插值 =============== 
            n_steps = 5
            
            for i in range(n_steps):
                
                style_vectors_interpolated = style_vectors1 + (i+1) * (style_vectors2-style_vectors1) / n_steps
                style_codes_interpolated = self.net.cal_style_codes(style_vectors_interpolated)
                intermediate_result1, _ , _ = self.net.gen_img(struc_code1, style_codes_interpolated, onehot1)

                intermediate_result1_vis = torch_utils.tensor2im(intermediate_result1[0])
                intermediate_result1_vis.save(os.path.join(self.opts.save_dir,"%s_to_%s_%d.png" % (img1_name,img2_name,i) ))    
                
            cnt += 1
            
            if cnt> num_pairs:
                break
            
    @torch.no_grad()
    def recon(self, indices=[],save_mask=False):
        
        for idx in indices:
            
            img1_name = os.path.basename(self.test_ds.imgs[idx]).split(".")[0]
            img1, mask1, mask_vis1 = self.test_ds[idx]
            
            img1, mask1 = img1.unsqueeze(0), mask1.unsqueeze(0)
            img1 = img1.to(self.opts.device).float()
            mask1 = (mask1*255).long().to(self.opts.device)
            # [bs,1,H,W]的mask 转成one-hot格式,即[bs,#seg_cls,H,W]
            onehot1 = torch_utils.labelMap2OneHot(mask1, num_cls=self.opts.num_seg_cls)
            
            # 原始的style codes
            style_vectors1, struc_code1 = self.net.get_style_vectors(img1, onehot1)
            style_codes1 = self.net.cal_style_codes(style_vectors1)
            
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
            noise = [torch.randn(1,512,4,4).to(self.opts.device)]
            for i in [8,16,32,64,128,256,512,1024]:
                noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
                noise.append(torch.randn(1,channels[i],i,i).to(self.opts.device))
            
            ## ==========  重建后的图片  ==========  
            recon1, _, structure_feats_1 = self.net.gen_img(struc_code1, style_codes1, onehot1,
                                                            randomize_noise=False, noise=noise)
            ## =======================================
            
            # 保存结果到图片
            vis_dict = {}
            vis_dict["img1"] = torch_utils.tensor2im(img1[0])
            
            # vis_dict["mask1"] = torch_utils.tensor2map(onehot1[0])
            
            if save_mask:
                vis_dict["edited_mask1"] = torch_utils.tensor2map(onehot1[0],shown_mask_indices=None)
                
            vis_dict["recon1"] = torch_utils.tensor2im(recon1[0])
                   
            self.save_imgs(vis_dict, save_name="%s" % img1_name)
    
    @torch.no_grad()
    def global_editing(self, indices=[], edit_type="age",degree=3):
        global_edit_direction = torch.load("./directions/%s.pt"%edit_type).to(self.opts.device)
        
        for idx in indices:    
            img1_name = os.path.basename(self.test_ds.imgs[idx]).split(".")[0]
            img1, mask1, mask_vis1 = self.test_ds[idx]
            
            img1, mask1 = img1.unsqueeze(0), mask1.unsqueeze(0)
            img1 = img1.to(self.opts.device).float()
            mask1 = (mask1*255).long().to(self.opts.device)
            # [bs,1,H,W]的mask 转成one-hot格式,即[bs,#seg_cls,H,W]
            onehot1 = torch_utils.labelMap2OneHot(mask1, num_cls=self.opts.num_seg_cls)
            
            # 原始的style codes
            style_vectors1, struc_code1 = self.net.get_style_vectors(img1, onehot1)
            style_codes1 = self.net.cal_style_codes(style_vectors1)
    
            ## ========== 重建 ==========  
            recon1, _, _ = self.net.gen_img(struc_code1, style_codes1, onehot1)
            
            ## ========== 编辑 ==========  
            # edit_style_codes1 = [style_codes1 + global_edit_direction * factor for factor in range(-degree,degree+1, 1)]
            # edits = [self.net.gen_img(struc_code1, edit_style_code, onehot1)[0]for edit_style_code in edit_style_codes1]
            
                    
            # 保存结果到图片
            vis_dict = {}
            vis_dict["img"] = torch_utils.tensor2im(img1[0])    
            vis_dict["recon"] = torch_utils.tensor2im(recon1[0])
            # for i in range(len(edits)):
            #     vis_dict["edit_%d"%i] = torch_utils.tensor2im(edits[i][0])
                   
            self.save_imgs(vis_dict, save_name="%s" % img1_name)
            
            
if __name__ == '__main__':
    opts = EditOptions().parse()
    os.makedirs(opts.save_dir, exist_ok=True)
    editor = Editor(opts)
    editor.edit(num_pairs=1,edit_mode='comp_style')
    # editor.interpolation(num_pairs=20)
    