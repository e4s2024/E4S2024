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
from training.video_swap_ft_coach import VideoSwapPTICoach
# from training.video_swap_st_constraint import VideoSwapPTICoach
# from training.video_swap_stich_coach import VideoSwapStichingCoach
from tqdm import trange
from options.our_swap_face_pipeline_options import OurSwapFacePipelineOptions
from swap_face_fine.swap_face_mask import swap_head_mask_revisit, swap_head_mask_hole_first
from utils.morphology import dilation, erosion
from training.video_swap_ft_coach import dialate_mask, erode_mask
from swap_face_fine.multi_band_blending import blending

def create_masks(mask, operation='dilation', radius=0):
    temp = copy.deepcopy(mask)
    if operation == 'dilation':
        full_mask = dilation(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = full_mask - temp
    elif operation == 'erosion':
        full_mask = erosion(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = temp - full_mask
    # 'expansion' means to obtain a boundary that expands to both sides
    elif operation == 'expansion':
        full_mask = dilation(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        erosion_mask = erosion(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = full_mask - erosion_mask

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
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=0.3*((kernel_size[0]-1)*0.5-1)+0.8)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    pasted_image.alpha_composite(image_masked)
    return pasted_image

# ===================================
    
def crop_and_align_face(target_files, image_size = 1024, scale = 1.0, center_sigma = 1.0, xy_sigma = 3.0, use_fa = False):
    
    from utils.alignment import crop_faces, calc_alignment_coefficients # 这一句有 import的 bug，大概率 dlib的问题
    
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
        
    # 额外处理一下耳朵 和 耳环部分
    
    # 如果 source 没有耳朵，那么就用 source 和target 耳朵style vectors的平均值 (为了和皮肤贴合)
    # if torch.sum(style_vectors2[:,7,:]) == 0:
    style_vectors[:,7,:] = (style_vectors1[:,7,:] + style_vectors2[:,7,:]) / 2
    
    # 直接用 target 耳环的style vector
    style_vectors[:,11,:] = style_vectors1[:,11,:] 
    
    # 脖子用二者的插值
    if belowFace_interpolation:
        style_vectors[:,8,:] = (style_vectors1[:,8,:] + style_vectors2[:,8,:]) / 2
    
    # 如果source 没有牙齿，那么用 target 牙齿的 style vector
    if torch.sum(style_vectors2[:,9,:]) == 0:
        style_vectors[:,9,:] = style_vectors1[:,9,:] 
        
    return style_vectors
    

    
# 图片换脸整个流程的前2步, 输入的人脸可以是任意图片
def faceSwapping_pipeline_step12(source, target, opts, need_crop = False):
    """
    Refine 后的换脸的流程:

        输入: target image, source image, 模型 G

        (1) 将 target image 和 source image 分别 crop, 并对齐, 得到 S 和T ; (crop是可选项)
        (2) 用 faceVid2Vid + GPEN, 将 S 按照T进行驱动, 得到driven image (D), 并提取D 的 mask ;
        
        (3) 将 D 送到模型G, 得到初始的 style vectors, 并随后开始按照 PTI 模式优化模型, 得到优化后的模型 G' , 
            这一步 finetune是为了让 style code和驱动的结果尽可能的相似, 并且去除脸部的抖动
            
        (4) 交换 D 和 T 的脸部的 style vector, 交换 D 和 T 的脸部mask;
        
        (5) 将编辑好的 mask, 以及交换后的 style vectors, 以及模型G', 实现换脸, 得到结果图片 I; 
            注意, 这一步要再次finetune 模型, 优化的目标是 inner face 和第(4)步的inner face一样, 且在dilated border部分和 target 的背景一样, 得到模型G''

        (6) 将 I 贴回到 target image中, 这一步可以借助 dilation mask, 以及 Gaussian blurring;

    Args:
        source (str):
        target (str):
    """
    from skimage.transform import resize  # 这一句有点问题
    from swap_face_fine.face_vid2vid.drive_demo import init_facevid2vid_pretrained_model, drive_source_demo  # 这一行有点问题
    from swap_face_fine.gpen.gpen_demo import init_gpen_pretrained_model, GPEN_demo
    from swap_face_fine.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps
    # ================= 加载模型相关 =========================
    # face_vid2vid 模型
    face_vid2vid_cfg = "./pretrained/faceVid2Vid/vox-256.yaml"
    face_vid2vid_ckpt = "./pretrained/faceVid2Vid/00000189-checkpoint.pth.tar"
    generator, kp_detector, he_estimator, estimate_jacobian = init_facevid2vid_pretrained_model(face_vid2vid_cfg, face_vid2vid_ckpt)
    # GPEN 模型
    GPEN_model = init_gpen_pretrained_model()  # grad bug
    
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
    
    save_dir = os.path.join(opts.exp_dir, "intermediate_results")
    os.makedirs(save_dir, exist_ok = True)
    os.makedirs(os.path.join(save_dir, "imgs"), exist_ok = True)
    os.makedirs(os.path.join(save_dir, "mask"), exist_ok = True)
    os.makedirs(os.path.join(save_dir, "styleVec"), exist_ok = True)
    
    source_files = [source]
    source_files = [(os.path.basename(f).split('.')[0], f) for f in source_files] # 只有1张
    
    target_files = target
    target_files = [(os.path.basename(f).split('.')[0], f) for f in target_files] # 很多张

    ret_dict = {}

    # (1) 将 target image 和 source image 分别 crop, 并对齐, 得到 S 和T
    if need_crop:
        target_crops, target_orig_images, target_quads, target_inv_transforms = crop_and_align_face(
            target_files, image_size = 1024, scale = 1.0, center_sigma = 1.0, xy_sigma = 3.0, use_fa = False
        )
        T = [crop.convert("RGB") for crop in target_crops]
        
        source_crops, source_orig_images, source_quads, source_inv_transforms = crop_and_align_face(
            source_files, image_size = 1024, scale = 1.0, center_sigma = 0, xy_sigma = 0, use_fa = False
        )
        S = source_crops[0].convert("RGB")

        ret_dict["t_ori"] = target_orig_images
        ret_dict["t_inv_trans"] = target_inv_transforms
        
    else:
        S = Image.open(source).convert("RGB").resize((1024, 1024))
        T = [Image.open(t).convert("RGB").resize((1024, 1024)) for t in target]
    
    S_mask = faceParsing_demo(faceParsing_model, S, convert_to_seg12=True)
    T_mask = [faceParsing_demo(faceParsing_model, frm, convert_to_seg12=True) for frm in T]
    Image.fromarray(S_mask).save(os.path.join(save_dir,"mask","S_mask.png"))
    for i in range(len(T_mask)):
        T[i].save(os.path.join(save_dir,"imgs","T_%04d.png"%i))
        Image.fromarray(T_mask[i]).save(os.path.join(save_dir,"mask","T_mask_%04d.png"%i))
    
    # 256,[0,1]范围
    S_256 = resize(np.array(S)/255.0, (256, 256))
    T_256 = [resize(np.array(im)/255.0, (256, 256)) for im in T]
    
    # (2) 1. 用 faceVid2Vid, 将 S 按照T进行驱动, 输入[0,1]范围RGB顺序, 输出是[0,1]范围RGB顺序
    predictions = drive_source_demo(S_256, T_256, generator, kp_detector, he_estimator, estimate_jacobian)
    predictions = [(pred*255).astype(np.uint8) for pred in predictions]
    del generator, kp_detector, he_estimator
    
    # (2) 2. 用GPEN增强，得到driven image (D), 输入是[0,255]范围 BGR顺序, 输出是[0,255]范围 BGR顺序
    drivens = [GPEN_demo(pred[:,:,::-1], GPEN_model, aligned=False) for pred in predictions]
    # 将 GPEN [0,255]范围 BGR顺序的输出转为 PIL.Image
    D = [Image.fromarray(drivens[i][:,:,::-1]) for i in range(len(drivens))] 
    for i in range(len(T_mask)):
        D[i].save(os.path.join(save_dir,"imgs","D_%04d.png"%i))
        
    # (2) 3. 提取 driven D 的 mask
    D_mask = [faceParsing_demo(faceParsing_model, d, convert_to_seg12=True) for d in D] 
    for i in range(len(T_mask)):
        Image.fromarray(D_mask[i]).save(os.path.join(save_dir,"mask","D_mask_%04d.png"%i))
        D_mask_vis = vis_parsing_maps(D[i], D_mask[i]) 
        Image.fromarray(D_mask_vis).save(os.path.join(save_dir,"mask","D_mask_vis_%04d.png"%i))

    # 计算出初始化的 style vectors
    with torch.no_grad():
        for i, (d, t) in enumerate(zip(D, T)):
            # wrap data 
            driven = transforms.Compose([TO_TENSOR, NORMALIZE])(d)
            driven = driven.to(opts.device).float().unsqueeze(0)
            driven_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(D_mask[i]))
            driven_mask = (driven_mask*255).long().to(opts.device).unsqueeze(0)
            driven_onehot = torch_utils.labelMap2OneHot(driven_mask, num_cls = opts.num_seg_cls)

            target = transforms.Compose([TO_TENSOR, NORMALIZE])(t)
            target = target.to(opts.device).float().unsqueeze(0)
            target_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(T_mask[i]))
            target_mask = (target_mask*255).long().to(opts.device).unsqueeze(0)
            target_onehot = torch_utils.labelMap2OneHot(target_mask, num_cls = opts.num_seg_cls)

            driven_style_vector, _ = net.get_style_vectors(driven, driven_onehot)
            torch.save(driven_style_vector, os.path.join(save_dir,"styleVec","D_style_vec_%04d.pt"%i))
            
            # driven_style_codes = net.cal_style_codes(driven_style_vector)
            # driven_face, _ , structure_feats = net.gen_img(torch.zeros(1,512,32,32).to(opts.device), driven_style_codes, driven_onehot)                
            # driven_face_image = torch_utils.tensor2im(driven_face[0])
            # driven_face_image.save(os.path.join(save_dir,"D_recon_%04d.png"%i))
            
            target_style_vector, _ = net.get_style_vectors(target, target_onehot)
            torch.save(target_style_vector, os.path.join(save_dir,"styleVec","T_style_vec_%04d.pt"%i))
            
            # target_style_codes = net.cal_style_codes(target_style_vector)
            # target_face, _ , structure_feats = net.gen_img(torch.zeros(1,512,32,32).to(opts.device), target_style_codes, target_onehot)                
            # target_face_image = torch_utils.tensor2im(target_face[0])
            # target_face_image.save(os.path.join(save_dir,"T_recon_%04d.png"%i))

    return ret_dict


def faceSwapping_pipeline_step34(opts, n_targets, step12_result: dict = None):
    
    # (3) 将 D 送到模型G, 得到初始的 style vectors, 并随后开始按照 PTI 模式优化模型, 得到优化后的模型 G' ;
    if opts.max_pti_steps > 0:
        finetune_coach = VideoSwapPTICoach(opts, num_targets = n_targets, erode = True)
        finetune_coach.train()

        # 保存优化后的模型
        save_dict = finetune_coach.get_save_dict()
        torch.save(save_dict, os.path.join(opts.exp_dir, "finetuned_G_lr%f_iters%d.pth"%(opts.pti_learning_rate, opts.max_pti_steps)))
        
        # 优化后的模型， 重建driven的结果
        # finetune_coach.recon_driven()
        
        net = finetune_coach.net 
    else:
        net = Net3(opts)
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = net.to(opts.device).eval()
        
        ckpt_dict = torch.load(opts.PTI_checkpoint_path)
        net.latent_avg = ckpt_dict['latent_avg'].to(opts.device)
        net.load_state_dict(torch_utils.remove_module_prefix(ckpt_dict["state_dict"],prefix="module."))
        # net.load_state_dict(ckpt_dict["state_dict"])
        print("Load pre-trained model success! No need to finetune!")
    
    swap_save_dir = 'results'
    os.makedirs(os.path.join(opts.exp_dir, 'results'), exist_ok=True)
    t_inv_trans = step12_result["t_inv_trans"]
    t_ori = step12_result["t_ori"]
    # (4) 交换 D 和 T 的脸部mask
    for i in range(n_targets):
        T = Image.open(os.path.join(opts.exp_dir, "intermediate_results", "imgs", "T_%04d.png"%i)).convert("RGB").resize((1024, 1024))
        
        D_mask = np.array(Image.open(os.path.join(opts.exp_dir, "intermediate_results","mask", "D_mask_%04d.png"%i)))
        # D_mask, _ = dialate_mask(D_mask, Image.open("/apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/v_18_video_swapping/musk_to_874/intermediate_results/imgs/D_%04d.png"%i), radius=3, verbose=True)
        
        T_mask = np.array(Image.open(os.path.join(opts.exp_dir, "intermediate_results","mask", "T_mask_%04d.png"%i)))
        # T_mask, _ = erode_mask(T_mask, T, radius=1, verbose=True)
        
        # swapped_msk, hole_map, eyebrows_line = swap_head_mask_revisit(D_mask, T_mask) # 换头   
        swapped_msk, hole_mask, hole_map, eye_line = swap_head_mask_hole_first(D_mask, T_mask)     
        cv2.imwrite(os.path.join(opts.exp_dir,"intermediate_results", "mask", "swappedMask_%04d.png"%i), swapped_msk)

        swappped_one_hot = torch_utils.labelMap2OneHot(torch.from_numpy(swapped_msk).unsqueeze(0).unsqueeze(0).long(), num_cls=12).to(opts.device)
        # torch_utils.tensor2map(swappped_one_hot[0]).save(os.path.join(opts.exp_dir,"swappedMaskVis.png"))
    
        # 保留 trgt_style_vectors 的background, hair, 其余的全用 driven_style_vectors
        D_style_vector = torch.load(os.path.join(opts.exp_dir, "intermediate_results", "styleVec","D_style_vec_%04d.pt"%i)).to(opts.device).float()
        T_style_vector = torch.load(os.path.join(opts.exp_dir, "intermediate_results", "styleVec","T_style_vec_%04d.pt"%i)).to(opts.device).float()
        comp_indices = set(range(opts.num_seg_cls)) - {0, 4, 11}  # 9 mouth
        swapped_style_vectors =  swap_comp_style_vector(T_style_vector, D_style_vector, list(comp_indices), belowFace_interpolation=False)
    
        # 保存交换后的 style vectors 到文件
        # torch.save(swapped_style_vectors, os.path.join(opts.exp_dir, "intermediate_results","styleVec", "swapped_style_vec_%04d.pt"%i))
        
        # swapped_style_vectors_np = swapped_style_vectors.cpu().numpy()
        # np.save(os.path.join(opts.exp_dir, "intermediate_results","styleVec", "swapped_style_vec_%04d.npy"%i), swapped_style_vectors_np)

        # finetune后的模型, 生成的换脸结果, 目的stiching 优化时有内脸的约束
        # TODO
        with torch.no_grad():
            swapped_style_codes = net.cal_style_codes(swapped_style_vectors)
            swapped_face, _ , structure_feats = net.gen_img(torch.zeros(1,512,32,32).to(opts.device), swapped_style_codes, swappped_one_hot)                
            swapped_face_image = torch_utils.tensor2im(swapped_face[0])
            
            swapped_face_image = swapped_face_image.resize((512,512)).resize((1024,1024))
            swapped_m = transforms.Compose([TO_TENSOR])(swapped_msk)
            swapped_m = (swapped_m*255).long().to(opts.device).unsqueeze(0)

            # swapped_face_image.save(os.path.join(opts.exp_dir, "intermediate_results","imgs", "Swapped_after_PTI_%04d.png"%i))

            outer_dilation = 5  # 这个值可以调节
            mask_bg = logical_or_reduce(*[swapped_m == clz for clz in [0, 11, 7, 4, 8]])   # 4,8,7  # 如果是视频换脸，考虑把头发也弄进来当做背景的一部分, 11 earings 4 hair 8 neck 7 ear
            is_foreground = torch.logical_not(mask_bg)
            hole_index = hole_mask[None][None]
            is_foreground[hole_index[None]] = True
            foreground_mask = is_foreground.float()

            # foreground_mask = dilation(foreground_mask, torch.ones(2 * outer_dilation + 1, 2 * outer_dilation + 1, device=foreground_mask.device), engine='convolution')
            content_mask, border_mask, full_mask = create_masks(foreground_mask, operation='expansion', radius=5)

            # past back
            content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
            content_mask = content_mask[0, 0, :, :, None].cpu().numpy()
            border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
            border_mask = border_mask[0, 0, :, :, None].cpu().numpy()
            border_mask = np.repeat(border_mask, 3, axis=-1)

            swapped_and_pasted = swapped_face_image * content_mask + T * (1 - content_mask)
            # swapped_and_pasted = Image.fromarray(np.uint8(swapped_and_pasted))
            swapped_and_pasted = Image.fromarray(blending(np.array(T), swapped_and_pasted, mask=border_mask))
            pasted_image = swapped_and_pasted

            ## Gaussian blending with mask
            # 处理一下mask， 得到face 区域和 dialation区域
            '''
            outer_dilation = 5
            mask_bg_and_hair = logical_or_reduce(*[swapped_m == clz for clz in [0,4,8,11,7]])  # 4头发, 7耳朵，8脖子，
            is_foreground = torch.logical_not(mask_bg_and_hair)
            foreground_mask = is_foreground.float()

            content_mask, border_mask, full_mask = create_masks(foreground_mask, operation='dilation', radius=outer_dilation)

            content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
            border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
            full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=False)
            full_mask_image = Image.fromarray(255*full_mask[0,0,:,:].cpu().numpy().astype(np.uint8))

            # 贴回去
            pasted_image = smooth_face_boundry(swapped_face_image, T, full_mask_image, radius = outer_dilation)

            mask_face = logical_or_reduce(*[swapped_m == item for item in [1, 2, 3, 5, 6, 9]])
            _, face_border, _ = create_masks(mask_face.float(), operation='expansion', radius=5)
            # Erase the border below the eyebrows to get the hairline
            hairline_mask = face_border
            hairline_mask[:, :, eye_line-20:, :] = 0
            hairline_mask = F.interpolate(hairline_mask, (1024, 1024), mode='bilinear', align_corners=False)
            hairline_mask = hairline_mask[0, 0, :, :, None].cpu().numpy()
            hairline_mask = np.repeat(hairline_mask, 3, axis=-1)

            pasted_image = pasted_image.convert('RGB')
            pasted_image = Image.fromarray(blending(np.array(T), np.array(pasted_image), mask=hairline_mask))
            '''

            """ op1. directly paste """
            # pasted_image.save(os.path.join(opts.exp_dir, swap_save_dir, "swap_face_%04d.png"%i))
            """ op2. paste back """
            swapped_and_pasted = swapped_and_pasted.convert('RGBA')
            pasted_image = t_ori[i].convert('RGBA')
            swapped_and_pasted.putalpha(255)
            projected = swapped_and_pasted.transform(t_ori[i].size, Image.PERSPECTIVE, t_inv_trans[i],
                                                     Image.BILINEAR)
            pasted_image.alpha_composite(projected)
            pasted_image.save(os.path.join(opts.exp_dir, swap_save_dir, "swap_face_%04d.png" % i))
    
    
if __name__=="__main__":
    
    opts = OurSwapFacePipelineOptions().parse()
    
    opts.max_pti_steps = 0
    opts.exp_dir = './video_outputs/'

    os.makedirs(opts.exp_dir, exist_ok=True)
    
    # opts.PTI_checkpoint_path = '/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video_res/swap_taylor_to_target2/finetuned_G_lr0.001000_iters200.pth'
    opts.PTI_checkpoint_path = "./video_outputs/finetuned_G_lr0.001000_iters80.pth"

    # 前2步
    # n_targets = len(glob.glob(opts.exp_dir + '/intermediate_results/imgs/*.png')) // 2
    # print(n_targets)
    n_targets = 60
    
    target_frames = sorted(glob.glob(os.path.join("/home/yuange/Documents/E4S_v2/figs/video_infer", "tmp_frames/*.png")))[:n_targets]
    source = "/home/yuange/Documents/E4S_v2/figs/video_infer/source.png"
    
    # 12步
    res12 = faceSwapping_pipeline_step12(source, target_frames, opts, need_crop=True)
    print("[main] step 1,2 finished")

    # 34步
    faceSwapping_pipeline_step34(opts, n_targets=n_targets, step12_result=res12)
    