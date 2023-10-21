import copy
import cv2
import matplotlib.pyplot as plt
from datasets.dataset import CelebAHQDataset, get_transforms
from PIL import Image
import os.path as osp
import torch
from argparse import Namespace
import numpy as np
from utils.torch_utils import saveTensorToFile, readImgAsTensor
from models.networks import Net3
from options.train_options import TrainOptions
import torchvision.transforms as transforms
from datasets.dataset import CelebAHQDataset, get_transforms, TO_TENSOR, NORMALIZE, MASK_CONVERT_TF, FFHQDataset, FFHQ_MASK_CONVERT_TF, MASK_CONVERT_TF_DETAILED, __celebAHQ_masks_to_faceParser_mask_detailed
from utils import torch_utils
import os
from tqdm import tqdm
from torch.nn import functional as F
import glob
import random
import torch.nn as nn
from headpose.detect import PoseEstimator
import math

import imageio
from torch.utils.data import Dataset
from skimage.transform import resize
from skimage import img_as_ubyte
from criteria.id_loss import IDLoss
from criteria.face_parsing.face_parsing_loss import FaceParsingLoss
from criteria.lpips.lpips import LPIPS
from collections import defaultdict
from tqdm import trange
from options.our_swap_face_pipeline_options import OurSwapFacePipelineOptions
from swap_face_fine.color_transfer import skin_color_transfer
from swap_face_fine.multi_band_blending import blending

from swap_face_fine.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps

from swap_face_fine.swap_face_mask import swap_head_mask_revisit, swap_head_mask_revisit_considerGlass, swap_head_mask_hole_first, swap_head_mask_target_bg_dilation

from utils.alignment import crop_faces, calc_alignment_coefficients
from utils.morphology import dilation, erosion, opening
from utils.util import save, get_5_from_98, get_detector, get_lmk
from swap_face_fine.MISF.inpainting import inpainting_face
from swap_face_fine.deformation_demo import image_deformation

# from PIPNet.lib.tools import get_lmk_model, demo_image
from alignment import norm_crop, norm_crop_with_M


# ================= 加载模型相关 =========================
# face parsing 模型
faceParsing_ckpt = "./pretrained/faceseg/79999_iter.pth"
faceParsing_model = init_faceParsing_pretrained_model(faceParsing_ckpt)

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

def paste_image_mask(inverse_transform, image, dst_image, mask, hairline_mask=None, radius=0, sigma=0.0):

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
        image_masked.putalpha(1)
    

    projected = image_masked.transform(dst_image.size, Image.PERSPECTIVE, inverse_transform, Image.BILINEAR)
    pasted_image.alpha_composite(projected)

    return pasted_image

def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image


def smooth_face_boundry(image, dst_image, mask, radius=0, sigma=0.0):
    # 把 image 贴到 dst_image 上去, 其中mask是 image内脸对应的mask
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask) # mask 需要是 [0,255] 范围
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)  # border 填0可能会导致图片的边界处mask 不太对
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        image_masked.putalpha(mask)

    pasted_image.alpha_composite(image_masked)
    return pasted_image


def mouth_transfer(source, target, mouth_mask):

    mouth_mask = F.interpolate(mouth_mask, size=1024, mode='bilinear').clamp_(0, 1)
    mouth_mask = (mouth_mask > 0).float()
    _, seam_mask, _ = create_masks(mouth_mask, operation='expansion', radius=5)
    seam_mask = seam_mask.cpu().numpy()[0][0][:, :, None]
    seam_mask = np.repeat(seam_mask, 3, axis=-1)

    mouth_mask = mouth_mask.cpu().numpy()[0][0][:, :, None]
    mouth_mask = np.repeat(mouth_mask, 3, axis=-1)

    combined = np.array(source) * mouth_mask + np.array(target) * (1 - mouth_mask)
    combined = Image.fromarray(blending(np.array(source), combined, mask=seam_mask))
    return combined, Image.fromarray(np.uint8(mouth_mask * 255)), Image.fromarray(np.uint8(seam_mask * 255))


# ===================================     
def crop_and_align_face(target_files):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False
    
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
    """
    if torch.sum(style_vectors2[:,7,:]) == 0:
        style_vectors[:,7,:] = (style_vectors1[:,7,:] + style_vectors2[:,7,:]) / 2   # 这里直接插值
    """
    
    # # 如果 source 没有耳环，那么直接用 target 耳环的style vector
    # if torch.sum(style_vectors2[:,11,:]) == 0:
    #     style_vectors[:,11,:] = style_vectors1[:,11,:] 
    
    # 直接用 target 耳环的style vector
    style_vectors[:,11,:] = style_vectors1[:,11,:] 
    
    # 脖子用二者的插值
    if belowFace_interpolation:
        style_vectors[:,8,:] = (style_vectors1[:,8,:] + style_vectors2[:,8,:]) / 2
    
    # 如果source 没有牙齿，那么用 target 牙齿的 style vector
    if torch.sum(style_vectors2[:,9,:]) == 0:
        style_vectors[:,9,:] = style_vectors1[:,9,:] 
        
    return style_vectors
    
# 图片换脸整个流程, 输入的人脸可以是任意图片
@torch.no_grad()
def faceSwapping_pipeline(source, 
                            target, 
                            opts, 
                            net,
                            save_dir, 
                            target_mask=None, 
                            need_crop =True, 
                            verbose=True, 
                            only_target_crop=False, 
                            optimize_W=True,
                            finetune_net=False,
                            copy_face=True,
                            pose_drive='faceVid2Vid',
                            face_enhancement='gpen'
                            ):

    if not os.path.exists(save_dir) and verbose:
        os.makedirs(save_dir)

    source_and_target_files = [source, target]
    source_and_target_files = [(os.path.basename(f).split('.')[0], f) for f in source_and_target_files]
    result_name = "swap_%s_to_%s.png"%(source_and_target_files[0][0], source_and_target_files[1][0])

    # (1) 将 target image 和 source image 分别 crop, 并对齐, 得到 S 和T
    if only_target_crop:
        crops, orig_images, quads, inv_transforms = crop_and_align_face(source_and_target_files[1:])
        crops = [crop.convert("RGB") for crop in crops]
        T = crops[0]
        S = Image.open(source).convert("RGB").resize((1024, 1024))
    elif need_crop:
        crops, orig_images, quads, inv_transforms = crop_and_align_face(source_and_target_files)
        crops = [crop.convert("RGB") for crop in crops]
        S, T = crops
    else:
        S = Image.open(source).convert("RGB").resize((1024, 1024))
        T = Image.open(target).convert("RGB").resize((1024, 1024))
        crops = [S, T]

    S_256, T_256 = [resize(np.array(im)/255.0, (256, 256)) for im in [S, T]]  # 256,[0,1]范围
    T_mask = faceParsing_demo(faceParsing_model, T, convert_to_seg12=True) if target_mask is None else target_mask

    if verbose:
        Image.fromarray(T_mask).save(os.path.join(save_dir,"T_mask.png"))
        # T_mask_vis = vis_parsing_maps(T, T_mask)
        # Image.fromarray(T_mask_vis).save(os.path.join(save_dir,"T_mask_vis.png"))
        S.save(os.path.join(save_dir, "S_cropped.png"))
        T.save(os.path.join(save_dir, "T_cropped.png"))
    

    # ========================================= pose alignment ===============================================
    """
    est = PoseEstimator(weights='/apdcephfs_cq2/share_1290939/branchwang/pretrained_models/headpose/model_weights.zip')
    need_drive = False
    try:
        pose_s = est.pose_from_image(image=np.array(S))
        pose_t = est.pose_from_image(image=np.array(T))
        diff = (pose_s[0] - pose_t[0]) ** 2 + (pose_s[1] - pose_t[1]) ** 2 + (pose_s[2] - pose_t[2]) ** 2
        pose_threshold = 10.
        if math.sqrt(diff) > pose_threshold:
            need_drive = True
    except:
        need_drive = True
    """

    need_drive = False
    if need_drive:
        if pose_drive == 'TPSMM':
            from swap_face_fine.TPSMM.demo import drive_source_demo

            cfg_path = "/apdcephfs_cq2/share_1290939/branchwang/projects/Thin-Plate-Spline-Motion-Model/config/vox-256.yaml"
            ckpt_path = "/apdcephfs_cq2/share_1290939/branchwang/pretrained_models/Thin-Plate-Spline-Motion-Model/checkpoints/vox.pth.tar"

            driven = drive_source_demo(S_256, T_256, cfg_path=cfg_path, ckpt_path=ckpt_path)
            driven = (driven * 255).astype(np.uint8)
        elif pose_drive == 'PIRender':
            from Deep3DFaceRecon_pytorch.drive import drive_source_demo

            cfg_path = '/apdcephfs_cq2/share_1290939/branchwang/projects/PIRender/config/face_demo.yaml'
            deep3d_ckpt_path = "/apdcephfs_cq2/share_1290939/branchwang/projects/PIRender/Deep3DFaceRecon_pytorch/checkpoints"
            pirender_ckpt_path = '/apdcephfs_cq2/share_1290939/branchwang/projects/PIRender/result/face/epoch_00190_iteration_000400000_checkpoint.pt'

            driven = drive_source_demo(S, T, cfg_path, pirender_ckpt_path, deep3d_ckpt_path)
            driven = np.array(driven)
        elif pose_drive == 'faceVid2Vid':
            from swap_face_fine.face_vid2vid.drive_demo import init_facevid2vid_pretrained_model, drive_source_demo

            face_vid2vid_cfg = "/apdcephfs_cq2/share_1290939/branchwang/projects/One-Shot_Free-View_Neural_Talking_Head_Synthesis/config/vox-256.yaml"
            face_vid2vid_ckpt = "/apdcephfs_cq2/share_1290939/branchwang/projects/One-Shot_Free-View_Neural_Talking_Head_Synthesis/ckpts/00000189-checkpoint.pth.tar"

            generator, kp_detector, he_estimator, estimate_jacobian = init_facevid2vid_pretrained_model(face_vid2vid_cfg, face_vid2vid_ckpt)

            # 将 S 按照T进行驱动, 输入[0,1]范围RGB顺序, 输出是[0,1]范围RGB顺序
            predictions = drive_source_demo(S_256, [T_256], generator, kp_detector, he_estimator, estimate_jacobian)
            predictions = [(pred*255).astype(np.uint8) for pred in predictions]
            del generator, kp_detector, he_estimator
        
            # 用GPEN增强，得到driven image (D), 输入是[0,255]范围 BGR顺序, 输出是[0,255]范围 BGR顺序
            driven = predictions[0]
        elif pose_drive == 'DaGAN':
            from swap_face_fine.DaGAN.drive_demo import init_DaGAN_pretrained_model, drive_source_demo

            DaGAN_cfg = "/apdcephfs_cq2/share_1290939/branchwang/projects/CVPR2022-DaGAN/config/vox-adv-256.yaml"
            DaGAN_ckpt = "/apdcephfs_cq2/share_1290939/branchwang/projects/CVPR2022-DaGAN/ckpts/DaGAN_vox_adv_256.pth.tar"
            depth_path = "/apdcephfs_cq2/share_1290939/branchwang/projects/CVPR2022-DaGAN/ckpts/depth_face_model_Voxceleb2_10w"
            # depth_path = "/apdcephfs_cq2/share_1290939/branchwang/py_projects/CVPR2022-DaGAN/ckpts/depth_face_model"

            generator, kp_detector, depth_encoder, depth_decoder = init_DaGAN_pretrained_model(cfg_path=DaGAN_cfg, 
                                                                                        ckpt_path=DaGAN_ckpt, 
                                                                                        depth_path=depth_path, 
                                                                                        g='DepthAwareGenerator')

            predictions = drive_source_demo(S_256, [T_256], generator, kp_detector, depth_encoder, depth_decoder)
            predictions = [(pred * 255).astype(np.uint8) for pred in predictions]
        
            driven = predictions[0]
        elif pose_drive == 'LIA':
            from swap_face_fine.LIA.run_demo import drive_demo

            driven = drive_demo(S_256, T_256)
            driven = (driven * 255).astype(np.uint8)
        else:
            raise ValueError(f'Wrong pose drive mode {pose_drive}.')
    else:
        driven = (S_256 * 255).astype(np.uint8)

    if face_enhancement == 'gfpgan':
        from swap_face_fine.GFPGAN.inference_gfpgan import face_restoration
        driven = face_restoration([driven])
        D = Image.fromarray(driven[0])
    elif face_enhancement == 'gpen':
        from swap_face_fine.gpen.gpen_demo import init_gpen_pretrained_model, GPEN_demo

        GPEN_model = init_gpen_pretrained_model()

        driven = GPEN_demo(driven[:, :, ::-1], GPEN_model, aligned=False)
        D = Image.fromarray(driven[:,:,::-1])
    else:
        raise ValueError(f'Wrong face enhancement mode {face_enhancement}.')


    """
    # =========================== Warp the target face shape to the source =======================================
    inv_trans_coeffs, orig_image = inv_transforms[1], orig_images[1]
    source_trans = D.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
    target_warped = image_deformation(image=orig_image, image_ref=source_trans)
    crops, orig_images, quads, inv_transforms = crop_and_align_face([(source_and_target_files[0][0], source_trans), (source_and_target_files[1][0], target_warped)])
    crops = [crop.convert("RGB") for crop in crops]
    _, T = crops

    T_mask = faceParsing_demo(faceParsing_model, T, convert_to_seg12=True)
    if verbose:
        Image.fromarray(T_mask).save(os.path.join(save_dir,"T_mask_deformed.png"))
        T.save(os.path.join(save_dir, "T_cropped_deformed.png"))
    """

    # (2) 3. 提取 driven D 的 mask
    D_mask = faceParsing_demo(faceParsing_model, D, convert_to_seg12=True)

    if verbose:
        Image.fromarray(D_mask).save(os.path.join(save_dir,"D_mask.png"))
        D.save(os.path.join(save_dir,"D_%s_to_%s.png"%(source_and_target_files[0][0], source_and_target_files[1][0])))
        # D_mask_vis = vis_parsing_maps(D, D_mask)
        # Image.fromarray(D_mask_vis).save(os.path.join(save_dir,"D_mask_vis.png"))
        
    # driven_m_dilated, dilated_verbose = dilate_mask(D_mask, D , radius=3, verbose=True)

    driven_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(D_mask))
    driven_mask = (driven_mask*255).long().to(opts.device).unsqueeze(0)
    driven_onehot = torch_utils.labelMap2OneHot(driven_mask, num_cls = opts.num_seg_cls)
    target_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(T_mask))
    target_mask = (target_mask*255).long().to(opts.device).unsqueeze(0)
    target_onehot = torch_utils.labelMap2OneHot(target_mask, num_cls = opts.num_seg_cls)


    # ========================================= inner face area skin color transfer ==========================================
    """
    mask_face_driven = logical_or_reduce(*[driven_mask == item for item in [1, 2, 3, 5, 6, 9, 7, 8]]).float()
    mask_face_target = logical_or_reduce(*[target_mask == item for item in [1, 2, 3, 5, 6, 9, 7, 8]]).float()

    # radius = 10
    # mask_face_swapped = dilation(mask_face_swapped, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask_face_swapped.device), engine='convolution')
    # mask_face_target = dilation(mask_face_target, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask_face_target.device), engine='convolution')
    mask_face_driven = F.interpolate(mask_face_driven, (1024, 1024), mode='bilinear', align_corners=False)
    mask_face_target = F.interpolate(mask_face_target, (1024, 1024), mode='bilinear', align_corners=False)

    _, face_border, _ = create_masks(mask_face_driven, operation='expansion', radius=5)
    face_border = face_border[0, 0, :, :, None].cpu().numpy()
    face_border = np.repeat(face_border, 3, axis=-1)
    
    mask_face_driven = mask_face_driven[0, 0, :, :, None].cpu().numpy()
    mask_face_target = mask_face_target[0, 0, :, :, None].cpu().numpy()
    driven_face_inner = D * mask_face_driven
    target_face_inner = T * mask_face_target

    # skin color transfer
    # ct_mode: color transfer mode. 
    # options: ['lct', 'rct', 'mkl', 'idt', 'sot', 'mix', 'adaptive']
    # 'adaptive': https://github.com/wkcn/Adaptive-Fast-Face-Color-Transfer
    # Others from DeepFaceLab: https://github.com/iperov/DeepFaceLab/blob/master/core/imagelib/color_transfer.py
    driven_face_inner = Image.fromarray(np.uint8(skin_color_transfer(np.array(driven_face_inner) / 255., 
                                            np.array(target_face_inner) / 255., 
                                            ct_mode='sot')))
    
    # swapped_face_original = np.array(D)
    driven_face_image = D * (1 - mask_face_driven) + driven_face_inner * mask_face_driven
    # driven_face_image = Image.fromarray(np.uint8(driven_face_image))
    driven_face_image = Image.fromarray(blending(np.array(D), np.array(driven_face_image), mask=face_border))
    D = driven_face_image
    if verbose:
        driven_face_image.save(os.path.join(save_dir, "face_color_transfer.png"))
        driven_face_mask = logical_or_reduce(*[driven_mask == item for item in [1, 2, 3, 5, 6, 9]]).float()
        driven_face_mask_image = Image.fromarray((driven_face_mask[0, 0, :, :] * 255).cpu().numpy().astype(np.uint8))
        driven_face_mask_image.save(os.path.join(save_dir, 'driven_face_mask.png'))
    """

    # wrap data 
    driven = transforms.Compose([TO_TENSOR, NORMALIZE])(D)
    driven = driven.to(opts.device).float().unsqueeze(0)
    target = transforms.Compose([TO_TENSOR, NORMALIZE])(T)
    target = target.to(opts.device).float().unsqueeze(0)

    if verbose:
        torch_utils.tensor2map(driven_onehot[0]).save(os.path.join(save_dir,"D_mask_vis.png")) 
        torch_utils.tensor2map(target_onehot[0]).save(os.path.join(save_dir,"T_mask_vis.png")) 
    
    # (3) 计算出初始化的 style vectors
    if optimize_W:
        from options.optim_options import OptimOptions
        from optimization import Optimizer

        optim_opts = OptimOptions().parse()
        optim_opts.ds_frac = 1.0
        optim_opts.id_lambda = 0.1
        optim_opts.face_parsing_lambda = 0.1
        optim_opts.W_steps = 200

        optimizer = Optimizer(optim_opts, net=net)

        with torch.set_grad_enabled(True):
            driven_style_vector = optimizer.optim_W_online(D, D_mask)
            target_style_vector = optimizer.optim_W_online(T, T_mask)

        import gc 
        del optimizer
        gc.collect()
    else:
        with torch.no_grad():
            driven_style_vector, _ = net.get_style_vectors(driven, driven_onehot) 
            target_style_vector, _ = net.get_style_vectors(target, target_onehot)
        
    if verbose:
        torch.save(driven_style_vector, os.path.join(save_dir,"D_style_vec.pt"))
        driven_style_codes = net.cal_style_codes(driven_style_vector)
        driven_face, _ , structure_feats = net.gen_img(torch.zeros(1,512,32,32).to(opts.device), driven_style_codes, driven_onehot)                
        driven_face_image = torch_utils.tensor2im(driven_face[0])
        driven_face_image.save(os.path.join(save_dir,"D_recon.png"))

        torch.save(target_style_vector, os.path.join(save_dir,"T_style_vec.pt"))
        target_style_codes = net.cal_style_codes(target_style_vector)
        target_face, _ , structure_feats = net.gen_img(torch.zeros(1,512,32,32).to(opts.device), target_style_codes, target_onehot)                
        target_face_image = torch_utils.tensor2im(target_face[0])
        target_face_image.save(os.path.join(save_dir,"T_recon.png"))

    # (4) 交换 D 和 T 的脸部mask
    # swapped_msk, hole_map, eyebrows_line = swap_head_mask_revisit_considerGlass(D_mask, T_mask) # 换头, hole_map是待填补的区域
    swapped_msk, hole_mask, hole_map, eye_line = swap_head_mask_hole_first(D_mask, T_mask)
    # swapped_msk = np.array(Image.open("/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/vis/swapped_edited.png"))
    if verbose:
        cv2.imwrite(os.path.join(save_dir,"swappedMask.png"), swapped_msk)
        swappped_one_hot = torch_utils.labelMap2OneHot(torch.from_numpy(swapped_msk).unsqueeze(0).unsqueeze(0).long(), num_cls=12)
        torch_utils.tensor2map(swappped_one_hot[0]).save(os.path.join(save_dir,"swappedMaskVis.png"))

        hole_one_hot = torch_utils.labelMap2OneHot(torch.from_numpy(hole_map).unsqueeze(0).unsqueeze(0).long(), num_cls=12)
        torch_utils.tensor2map(hole_one_hot[0]).save(os.path.join(save_dir,"hole_map.png"))
        # Image.fromarray(hole_map.astype(np.uint8)).save(os.path.join(save_dir, "hole_map.png"))
    
    # 保留 trgt_style_vectors 的background, hair, era_rings, eye_glass 其余的全用 driven_style_vectors
    comp_indices = set(range(opts.num_seg_cls)) - {0, 10, 4, 11  }  # 10 glass, 8 neck
    swapped_style_vectors =  swap_comp_style_vector(target_style_vector, driven_style_vector, list(comp_indices), belowFace_interpolation=False)
    if verbose:
        torch.save(swapped_style_vectors, os.path.join(save_dir,"swapped_style_vec.pt"))

    if finetune_net:
        from options.optim_options import OptimOptions
        from optimization import Optimizer

        optim_opts = OptimOptions().parse()
        optim_opts.ds_frac = 1.0
        optim_opts.id_lambda = 0.1
        optim_opts.face_parsing_lambda = 0.1
        optim_opts.finetune_steps = 200

        optimizer = Optimizer(optim_opts, net=net)
        with torch.set_grad_enabled(True):
            net = optimizer.finetune_net_online(target=D, mask=D_mask, style_vectors=driven_style_vector)

        import gc 
        del optimizer
        gc.collect()
    else:
        pass

    swapped_msk = Image.fromarray(swapped_msk).convert('L')
    swapped_msk = transforms.Compose([TO_TENSOR])(swapped_msk)
    swapped_msk = (swapped_msk*255).long().to(opts.device).unsqueeze(0)
    # swapped_msk = opening(swapped_msk, torch.ones(2 * 2 + 1, 2 * 2 + 1, device=swapped_msk.device), engine='convolution').long()
    swapped_onehot = torch_utils.labelMap2OneHot(swapped_msk, num_cls = opts.num_seg_cls)
    # 换脸的结果            
    swapped_style_codes = net.cal_style_codes(swapped_style_vectors)
    swapped_face, _ , structure_feats = net.gen_img(torch.zeros(1,512,32,32).to(opts.device), swapped_style_codes, swapped_onehot)                
    swapped_face_image = torch_utils.tensor2im(swapped_face[0])

    if verbose:
        swapped_face_image.save(os.path.join(save_dir, 'swapped_face.png'))

    mask_face = logical_or_reduce(*[swapped_msk == item for item in [1, 2, 3, 5, 6, 9]])
    # copy & paste
    if copy_face:
        # mouth, lip
        mouth_mask = torch.logical_or(swapped_msk == 1, swapped_msk == 9)
        # eyes
        # mouth_mask = torch.logical_or(mouth_mask, torch.logical_or(swapped_msk == 3, swapped_msk == 2))
        # mouth_mask = torch.from_numpy(mouth_mask)
        # mouth_mask = mouth_mask[None, None, :, :].float()

        radius = 5
        mouth_mask = erosion(mouth_mask.float(), torch.ones(2 * radius + 1, 2 * radius + 1, device=mouth_mask.device), engine='convolution')
        radius = 20
        mouth_mask = dilation(mouth_mask, torch.ones(2 * radius + 1, 2 * radius + 1, device=mouth_mask.device), engine='convolution')
        mouth_mask = dilation(mouth_mask, torch.ones(2 * radius + 1, 2 * radius + 1, device=mouth_mask.device), engine='convolution')

        driven_face_area = logical_or_reduce(*[driven_mask == item for item in [1, 2, 3, 5, 6, 9]])
        mouth_mask = torch.logical_and(mouth_mask, driven_face_area)
        mouth_mask = torch.logical_and(mouth_mask, mask_face).float()

        swapped_face_image, mouth_mask_image, seam_mask_image = mouth_transfer(source=D, target=swapped_face_image, mouth_mask=mouth_mask)
        if verbose:
            mouth_mask_image.save(os.path.join(save_dir, 'mouth_mask.png'))
            seam_mask_image.save(os.path.join(save_dir, 'mouth_seam_mask.png'))

    # inpainting
    """
    swapped_face_256, mask_256 = [resize(np.array(im), (256, 256)) for im in [swapped_face_image, hole_mask]]
    swapped_face_inpainting = inpainting_face(swapped_face_256 * 255, mask_256 * 255)
    swapped_face_inpainting = np.uint8(swapped_face_inpainting)
    swapped_face_up = GPEN_demo(swapped_face_inpainting[:,:,::-1], GPEN_model, aligned=False)
    swapped_face_image = Image.fromarray(swapped_face_up[:,:,::-1])

    swapped_msk = faceParsing_demo(faceParsing_model, swapped_face_image, convert_to_seg12=True)
    swapped_msk = transforms.Compose([TO_TENSOR])(Image.fromarray(swapped_msk))
    swapped_msk = (swapped_msk * 255).long().to(opts.device).unsqueeze(0)

    if verbose:
        swapped_face_image.save(os.path.join(save_dir, 'swapped_face_inpainting.png'))
        swapped_mask_inpainting_onehot = torch_utils.labelMap2OneHot(swapped_msk, num_cls = opts.num_seg_cls)
        torch_utils.tensor2map(swapped_mask_inpainting_onehot[0]).save(os.path.join(save_dir,"swapped_mask_inpainting.png"))
    """

    # (6) 最后贴回去
    outer_dilation = 5  # 这个值可以调节
    mask_bg = logical_or_reduce(*[swapped_msk == clz for clz in [0, 11, 4     ]])   # 4,8,7  # 如果是视频换脸，考虑把头发也弄进来当做背景的一部分, 11 earings 4 hair 8 neck 7 ear
    is_foreground = torch.logical_not(mask_bg)
    hole_index = hole_mask[None][None]
    is_foreground[hole_index[None]] = True
    foreground_mask = is_foreground.float()
            
    content_mask, border_mask, full_mask = create_masks(foreground_mask, operation='dilation', radius=outer_dilation)
        
    content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
    border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
    full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=False)
    full_mask_image = Image.fromarray(255*full_mask[0,0,:,:].cpu().numpy().astype(np.uint8))
            
    # 贴回去    
    pasted_image = smooth_face_boundry(swapped_face_image, T, full_mask_image, radius = outer_dilation)

    mask_face = logical_or_reduce(*[swapped_msk == item for item in [1, 2, 3, 5, 6, 9]])
    _, face_border, _ = create_masks(mask_face.float(), operation='expansion', radius=5)
    # Erase the border below the eyebrows to get the hairline
    hairline_mask = face_border
    hairline_mask[:, :, eye_line-20:, :] = 0
    hairline_mask = F.interpolate(hairline_mask, (1024, 1024), mode='bilinear', align_corners=False)
    hairline_mask = hairline_mask[0, 0, :, :, None].cpu().numpy()
    hairline_mask = np.repeat(hairline_mask, 3, axis=-1)

    pasted_image = pasted_image.convert('RGB')
    pasted_image = Image.fromarray(blending(np.array(T), np.array(pasted_image), mask=hairline_mask))

    return pasted_image
    
    
@torch.no_grad()
def interpolation(souece_name, target_name):
    T = Image.open("/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/images/%s.jpg"%target_name).convert("RGB").resize((1024, 1024))
    save_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/our_editing/tmp"
    result_name = "swap_%s_to_%s"%(souece_name, target_name)
    style_vectors1 =  torch.load("/apdcephfs/share_1290939/zhianliu/py_projects/our_editing/tmp/swapped_style_vec.pt")
    style_vectors2 =  torch.load("/apdcephfs/share_1290939/zhianliu/py_projects/our_editing/tmp/T_style_vec.pt")
    mask = torch.from_numpy(np.array(Image.open("/apdcephfs/share_1290939/zhianliu/py_projects/our_editing/tmp/swappedMask.png").convert("L")))
    
    mask = mask.unsqueeze(0).unsqueeze(0).long().to(opts.device)
    onehot = torch_utils.labelMap2OneHot(mask, num_cls=opts.num_seg_cls)
    
    # Gaussian blending with mask
    # 处理一下mask， 得到face 区域和 dialation区域
    outer_dilation = 1
    mask_bg = logical_or_reduce(*[mask == clz for clz in [0,11]])  # 如果是视频换脸，考虑把头发也弄进来当做背景的一部分
    is_foreground = torch.logical_not(mask_bg)
    foreground_mask = is_foreground.float()
    
    content_mask, border_mask, full_mask = create_masks(foreground_mask, outer_dilation = outer_dilation)
    
    content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
    content_mask_image = Image.fromarray(255*content_mask[0,0,:,:].cpu().numpy().astype(np.uint8))
    border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
    full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=False)
    full_mask_image = Image.fromarray(255*full_mask[0,0,:,:].cpu().numpy().astype(np.uint8))
        
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
        
    # ========== 开始插值 =============== 
    n_steps = 5
    
    for i in range(n_steps):
        
        style_vectors_interpolated = style_vectors1 + (i+1) * (style_vectors2-style_vectors1) / n_steps
        style_codes_interpolated = net.cal_style_codes(style_vectors_interpolated)
        intermediate_result, _ , _ = net.gen_img(torch.zeros(1,512,32,32, device=style_vectors1.device), style_codes_interpolated, onehot)

        intermediate_result_image = torch_utils.tensor2im(intermediate_result[0])

        # 直接贴，但是脸部和背景融合的地方 smooth一下
        if outer_dilation == 0:
            pasted_image = smooth_face_boundry(intermediate_result_image, T, content_mask_image, radius=outer_dilation)
        else:
            pasted_image = smooth_face_boundry(intermediate_result_image, T, full_mask_image, radius=outer_dilation)
        pasted_image.save(os.path.join(save_dir, "%s_%d.png"%(result_name,i+1)))

    
# 换脸 & 插值
"""
source_names = ["28688"]
target_names = ["28558"]
image_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/images"
label_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/labels"
for source_name, target_name in zip(source_names, target_names):
    source = os.path.join(image_dir, "%s.jpg"%source_name)
    target = os.path.join(image_dir, "%s.jpg"%target_name)
    target_mask = Image.open(os.path.join(label_dir, "%s.png"%target_name)).convert("L")
    target_mask_seg12 = __celebAHQ_masks_to_faceParser_mask_detailed(target_mask)      
    faceSwapping_pipeline(source, target, opts, save_dir="./tmp", target_mask = target_mask_seg12, need_crop = False, verbose = True) 
    interpolation(source_name, target_name)
"""