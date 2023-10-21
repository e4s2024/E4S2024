import copy
import cv2
from PIL import Image
import torch
import numpy as np
from models.networks import Net3
import torchvision.transforms as transforms
from datasets.dataset import get_transforms, TO_TENSOR, NORMALIZE
from utils import torch_utils
import os
from tqdm import tqdm
from torch.nn import functional as F
import glob
import torch.nn as nn
from tqdm import trange
from utils.morphology import dilation
from options.our_swap_face_pipeline_options import OurSwapFacePipelineOptions

regions_map = {'background':0, 'lip':1, 'eyebrows':2, 'eyes':3, 'hair':4,
               'nose':5, 'skin':6, 'ears':7, 'belowface':8,'mouth':9,
               'eye_glass':10,'ear_rings':11}


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
noise = [torch.randn(1,512,4,4).to("cuda:0")]
for i in [8,16,32,64,128,256,512,1024]:
    noise.append(torch.randn(1,channels[i],i,i).to("cuda:0"))
    noise.append(torch.randn(1,channels[i],i,i).to("cuda:0"))
                
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

def paste_image_mask(inverse_transform, image, dst_image, mask, radius=0, sigma=0.0):
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask)
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    projected = image_masked.transform(dst_image.size, Image.PERSPECTIVE, inverse_transform, Image.BILINEAR)
    pasted_image.alpha_composite(projected)
    return pasted_image

def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image


def smooth_face_boundry(image, dst_image, mask, radius=0, sigma=0.0):
    # 把 image 贴到 dst_image 上去, 其中mask是 image对应的mask
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask) # mask 需要是 [0,255] 范围
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    pasted_image.alpha_composite(image_masked)
    return pasted_image

# ===================================
    
def crop_and_align_face(target_files, image_size = 1024, scale = 1.0, center_sigma = 1.0, xy_sigma = 3.0, use_fa = False):
    
    from utils.alignment import crop_faces, calc_alignment_coefficients # 这一句有 import的 bug，大概率 face_alignment 的问题
    
    print('Aligning images')

    crops, orig_images, quads = crop_faces(image_size, target_files, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    
    # crop的逆变换，用于后期贴回到原始视频上去
    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]
    
    return crops, orig_images, quads, inv_transforms
                    
def swap_comp_style_vector(style_vectors1, style_vectors2, comp_indices=[], belowFace_interpolation=False):
    """替换 style_vectors1 中某个component的 style vectors

    Args:
        style_vectors1 (Tensor): with shape [1,#comp,512], target image 的 style vectors
        style_vectors2 (Tensor): with shape [1,#comp,512], source image 的 style vectors
    """
    assert comp_indices is not None
    
    style_vectors = copy.deepcopy(style_vectors1)
    
    for comp_idx in comp_indices:
        style_vectors[:,comp_idx,:] =  style_vectors2[:,comp_idx,:]
        
    # 脖子用二者的插值
    if belowFace_interpolation:
        style_vectors[:,8,:] = (style_vectors1[:,8,:] + style_vectors2[:,8,:]) / 2
    
    # 如果source 没有牙齿，那么用 target 牙齿的 style vector
    if torch.sum(style_vectors2[:,9,:]) == 0:
        style_vectors[:,9,:] = style_vectors1[:,9,:] 
        
    return style_vectors
    

def swap_head_mask(source, target, source_comp_indices=[]):
    res = np.zeros_like(target)
    target_regions = [np.equal(target, i) for i in range(12)]
    source_regions = [np.equal(source, i) for i in range(12)]
    
    
    tmp_set = set(range(12))
    # 先贴target的背景、belowface，
    res[target_regions[0]] = 99  # a magic number, 先占住背景区域，用99标记
    res[target_regions[8]] = 8
    [tmp_set.remove(k) for k in [0,8]]
    
    # 再贴source的皮肤
    res[source_regions[6]] = 6
    tmp_set.remove(6)
    
    for idx in source_comp_indices:
        res[source_regions[idx]] = idx
        tmp_set.remove(idx)
    
    for idx in list(tmp_set - {4,7,11}):     # 7-ear 4-hair 11-ear_rings
        res[target_regions[idx]] = idx
        tmp_set.remove(idx)
        
    # 再贴target的头发
    # 头发和背景的并集
    target_hair_bg = np.logical_or(np.equal(res,99), target_regions[4])
    res[target_hair_bg] = target[target_hair_bg]
    res[target_regions[0]] = 99
    tmp_set.remove(4)
    
    # 耳朵 以及 耳环
    res[target_regions[7]] = 7
    res[target_regions[11]] = 11
    tmp_set.remove(7)
    tmp_set.remove(11)
    
    assert len(tmp_set) == 0, "检测到有某些region未处理!"
    
    # 剩余是0的地方，我们填皮肤
    if np.sum(res==0) != 0:
        hole_map = 255*(res==0)
        res[res==0] = 6
    else:
        hole_map = np.zeros_like(res)
        
    # 最后把背景的label恢复一下
    res[res==99] = 0
     
    return res, hole_map

    
@torch.no_grad()
def video_editing(source, target, opts, both_crop = False, only_target_crop=False, verbose=False):
    """
    正脸视频换脸的流程:

        输入: target image, source image, 模型 G

        (1) 将 target image 和 source image 分别 crop, 并对齐, 得到 S 和T ; (crop是可选项)
        
        (2) 将 source 按照 target 进行驱动, 得到driven image;
        
        (3) 分别得到 driven 和 target的mask , 并分别提取 driven 和 target 的 style vector ;
        
        (4) 交换 D 和 T 的脸部的 style vector, 交换 D 和 T 的脸部mask;

        (5) 将 I 贴回到 target image中, 这一步可以借助 dilation mask, 以及 Gaussian blurring;

    Args:
        source (str): source 图片
        target (List[str]):  target video
        opts ():
        both_crop (bool): 是否需要将 source 和 target 全部 crop
        only_target_crop (bool): 只crop target video
        verbose (bool): 可视化中间的结果 (会占用大量的I/O时间) 

    """
    from skimage.transform import resize
    from swap_face_fine.face_vid2vid.drive_demo import init_facevid2vid_pretrained_model, drive_source_demo 
    from swap_face_fine.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps
    
    # ================= 加载模型相关 ========================= 
    # face_vid2vid 模型
    face_vid2vid_cfg = "/apdcephfs/share_1290939/zhianliu/py_projects/One-Shot_Free-View_Neural_Talking_Head_Synthesis/config/vox-256.yaml"
    face_vid2vid_ckpt = "/apdcephfs/share_1290939/zhianliu/py_projects/One-Shot_Free-View_Neural_Talking_Head_Synthesis/ckpts/00000189-checkpoint.pth.tar"
    generator, kp_detector, he_estimator, estimate_jacobian = init_facevid2vid_pretrained_model(face_vid2vid_cfg, face_vid2vid_ckpt)
    
    # face parsing 模型
    faceParsing_ckpt = "./pretrained/faceseg/79999_iter.pth"
    faceParsing_model = init_faceParsing_pretrained_model(faceParsing_ckpt)

    # 定义 我们的模型 G 和相关的 Loss func
    net = Net3(opts)
    net = net.to(opts.device)
    save_dict = torch.load(opts.checkpoint_path)
    net.load_state_dict(torch_utils.remove_module_prefix(save_dict["state_dict"],prefix="module."))
    net.latent_avg = save_dict['latent_avg'].to(opts.device)
    print("Load LocalStyleGAN pre-trained model success!") 
    # ========================================================
    
    # save_dir = os.path.join("/apdcephfs/share_1290939/zhianliu/py_projects/srcSkin_srcFaceShape_tgtMouth")
    save_dir = "/apdcephfs/share_1290939/paidaxingli/project/ft/haitong1"
    os.makedirs(save_dir, exist_ok = True)
    os.makedirs(os.path.join(save_dir, "imgs"), exist_ok = True)
    os.makedirs(os.path.join(save_dir, "results"), exist_ok = True)
    os.makedirs(os.path.join(save_dir, "mask"), exist_ok = True)
    os.makedirs(os.path.join(save_dir, "styleVec"), exist_ok = True)
    
    source_files = [source]
    source_files = [(os.path.basename(f).split('.')[0], f) for f in source_files] # 只有1张
    print (source_files)

    target_files = target
    target_files = [(os.path.basename(f).split('.')[0], f) for f in target_files] # 很多张
    print (target_files)


    # (1) 将 target image 和 source image 分别 crop, 并对齐, 得到 S 和T
    if only_target_crop:
        target_crops, target_orig_images, target_quads, target_inv_transforms = crop_and_align_face(
            target_files, image_size = 1024, scale = 1.0, center_sigma = 1.0, xy_sigma = 3.0, use_fa = False
        )
        T = [crop.convert("RGB") for crop in target_crops]
        S = Image.open(source).convert("RGB").resize((1024, 1024))
        
    elif both_crop:
        target_crops, target_orig_images, target_quads, target_inv_transforms = crop_and_align_face(
            target_files, image_size = 1024, scale = 1.0, center_sigma = 1.0, xy_sigma = 3.0, use_fa = False
        )
        T = [crop.convert("RGB") for crop in target_crops]
        
        source_crops, source_orig_images, source_quads, source_inv_transforms = crop_and_align_face(
            source_files, image_size = 1024, scale = 1.0, center_sigma = 0, xy_sigma = 0, use_fa = False
        )
        S = source_crops[0].convert("RGB")
        
    else:
        S = Image.open(source).convert("RGB").resize((1024, 1024))
        T = [Image.open(t).convert("RGB").resize((1024, 1024)) for t in target]
    
    S_mask = faceParsing_demo(faceParsing_model, S, convert_to_seg12=True)
    if verbose:
        S_mask_vis = vis_parsing_maps(S, S_mask) 
        Image.fromarray(S_mask_vis).save(os.path.join(save_dir,"mask","S_mask_vis.png"))
    
    
    # (2) 将 source 按照 target 进行驱动, 得到driven image;
    # 256,[0,1]范围
    S_256 = resize(np.array(S)/255.0, (256, 256))
    T_256 = [resize(np.array(im)/255.0, (256, 256)) for im in T]
    
    # 用 faceVid2Vid, 将 S 按照T进行驱动, 输入[0,1]范围RGB顺序, 输出是[0,1]范围RGB顺序
    predictions = drive_source_demo(S_256, T_256, generator, kp_detector, he_estimator, estimate_jacobian)
    predictions = [(pred*255).astype(np.uint8) for pred in predictions]
    del generator, kp_detector, he_estimator
    
    D = [Image.fromarray(predictions[i]).resize((1024,1024)) for i in range(len(predictions))] 
    if verbose:
        for i in range(len(D)):
            D[i].save(os.path.join(save_dir,"imgs","D_%04d.png"%i))
            
    # (3) 分别得到driven 和 target的mask , 并分别提取 driven 和 target 的 style vector 
   
    D_mask = [faceParsing_demo(faceParsing_model, frm, convert_to_seg12=True) for frm in D]    
    T_mask = [faceParsing_demo(faceParsing_model, frm, convert_to_seg12=True) for frm in T]
    
    if verbose:
        for i in range(len(D_mask)):
            D_mask_vis = vis_parsing_maps(D[i], D_mask[i]) 
            Image.fromarray(D_mask_vis).save(os.path.join(save_dir,"mask","D_mask_vis_%04d.png"%i))

            T_mask_vis = vis_parsing_maps(T[i], T_mask[i]) 
            Image.fromarray(T_mask_vis).save(os.path.join(save_dir,"mask","T_mask_vis%04d.png"%i))

            
    # # 计算出 source 的 style vectors
    # source = transforms.Compose([TO_TENSOR, NORMALIZE])(S)
    # source = source.to(opts.device).float().unsqueeze(0)
    # source_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(S_mask))
    # source_mask = (source_mask*255).long().to(opts.device).unsqueeze(0)
    # source_onehot = torch_utils.labelMap2OneHot(source_mask, num_cls = opts.num_seg_cls)
    
    # S_style_vector, _ = net.get_style_vectors(source, source_onehot)
    # if verbose:
    #     torch.save(S_style_vector, os.path.join(save_dir,"styleVec","S_style_vec.pt"))
    
    D_style_vectors = []
    T_style_vectors = []
    for i, (d,t) in enumerate(zip(D, T)):
        # wrap data 
        driven = transforms.Compose([TO_TENSOR, NORMALIZE])(d)
        driven = driven.to(opts.device).float().unsqueeze(0)
        driven_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(D_mask[i]))
        driven_mask = (driven_mask*255).long().to(opts.device).unsqueeze(0)
        driven_onehot = torch_utils.labelMap2OneHot(driven_mask, num_cls = opts.num_seg_cls)
        
        driven_style_vector, _ = net.get_style_vectors(driven, driven_onehot)
        D_style_vectors.append(driven_style_vector)
            
        target = transforms.Compose([TO_TENSOR, NORMALIZE])(t)
        target = target.to(opts.device).float().unsqueeze(0)
        target_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(T_mask[i]))
        target_mask = (target_mask*255).long().to(opts.device).unsqueeze(0)
        target_onehot = torch_utils.labelMap2OneHot(target_mask, num_cls = opts.num_seg_cls)
            
        target_style_vector, _ = net.get_style_vectors(target, target_onehot)
        T_style_vectors.append(target_style_vector)
        
              
    # (4) 交换 D 和 T 的脸部的 style vector, 交换 D 和 T 的脸部mask;
    for i in tqdm(range(len(T)),total=len(T)):
        
        swapped_msk, hole_map = swap_head_mask(D_mask[i], T_mask[i], source_comp_indices=[regions_map[key] for key in ["eyes", "eyebrows","nose","lip","mouth"]]) # 换头
        # swapped_msk, hole_map = swap_head_mask(D_mask, T_mask[i], source_comp_indices=[]) # 换头但保持target的五官形状        
        swappped_one_hot = torch_utils.labelMap2OneHot(torch.from_numpy(swapped_msk).unsqueeze(0).unsqueeze(0).long(), num_cls=12).to(opts.device)
        
        # comp_indices = set(range(opts.num_seg_cls)) - set([0,4,11]) - set([regions_map[key] for key in ["ears","skin","nose","belowface","mouth"]])  # 保持肤色
        comp_indices = set(range(opts.num_seg_cls)) - set([0,4,11]) # 肤色也换掉
        # comp_indices = set(range(opts.num_seg_cls)) - set([0,4,11]) - set([regions_map[key] for key in ["mouth"]]) # 肤色也换掉，同时使用target 的牙齿
        
        swapped_style_vectors =  swap_comp_style_vector(T_style_vectors[i], D_style_vectors[i], list(comp_indices), belowFace_interpolation=False)
    
        if verbose:
            torch_utils.tensor2map(swappped_one_hot[0]).save(os.path.join(save_dir, "mask", "swappedMaskVis_%04d.png"%i))
            # torch.save(swapped_style_vectors, os.path.join(save_dir,"styleVec", "swapped_style_vec_%04d.pt"%i))
                
        # (4) 生成的换脸结果
        with torch.no_grad():
            swapped_style_codes = net.cal_style_codes(swapped_style_vectors)
            swapped_face, _ , structure_feats = net.gen_img(torch.zeros(1,512,32,32).to(opts.device), swapped_style_codes, swappped_one_hot,
                                                            randomize_noise=False, noise=noise)                
            swapped_face_image = torch_utils.tensor2im(swapped_face[0])
            
            # swapped_face_image = swapped_face_image.resize((512,512)).resize((1024,1024))
            
            if verbose:
                swapped_face_image.save(os.path.join(save_dir, "results", "swapped_%04d.png"%i))
            
            ## Gaussian blending with mask
            # 处理一下mask， 得到face 区域和 dialation区域
            outer_dilation = 9
            swapped_m = transforms.Compose([TO_TENSOR])(swapped_msk)
            swapped_m = (swapped_m*255).long().to(opts.device).unsqueeze(0)
            mask_bg_and_hair = logical_or_reduce(*[swapped_m == clz for clz in [0,4,8,11,7]])  # 4头发, 7耳朵，8脖子，
            is_foreground = torch.logical_not(mask_bg_and_hair)
            foreground_mask = is_foreground.float()
            
            content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation = outer_dilation)
        
            content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
            border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
            full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=False)
            full_mask_image = Image.fromarray(255*full_mask[0,0,:,:].cpu().numpy().astype(np.uint8))
            
            # 贴回去    
            if both_crop or only_target_crop:
                pasted_image = paste_image_mask(target_inv_transforms[i], swapped_face_image, target_orig_images[i], full_mask_image, radius = outer_dilation)  
                pasted_image.save(os.path.join(save_dir,"results", "swapped_face_pasted_%04d.png"%i))
            else:
                pasted_image = smooth_face_boundry(swapped_face_image, T[i], full_mask_image, radius = outer_dilation)
                pasted_image.save(os.path.join(save_dir,"results", "swapped_face_%04d.png"%i))
    
    
if __name__=="__main__":
    target_frames = sorted(glob.glob(os.path.join("/apdcephfs/share_1290939/paidaxingli/project/ft/2/22","*.png")))
    source = sorted(glob.glob("/apdcephfs/share_1290939/paidaxingli/project/ft/ft_local/s1.jpg"))
    
    opts = OurSwapFacePipelineOptions().parse()
    for src in source:
        video_editing(src, target_frames, opts, both_crop = True, only_target_crop=False, verbose=False) 
    