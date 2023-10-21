
import numpy as np
import os
import json
import sys
import pprint
import random
import shutil
from PIL import Image
import glob
import copy
import torch
import cv2

from utils.morphology import dilation, erosion, opening

# CelebAMask-HQ原始的 18个属性,skin-1,nose-2,...cloth-18
# 额外的是0,表示背景
celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']

# face-parsing.PyTorch 虽然也用的是 CelebA-Mask-HQ 中的19个属性 ，但属性的顺序不太一样
FFHQ_label_list = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']  # skin-1 l_brow-2 ...

# 12个属性
faceParser_label_list_detailed = ['background', 'lip', 'eyebrows',
                         'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface','mouth','eye_glass','ear_rings']


def swap_head_mask_revisit(source, target):
    res = np.zeros_like(target)
    
    # 先找到target的 hair 区域 , belowface 区域, 耳朵区域
    target_hair = np.equal(target , 4)
    target_belowface = np.equal(target , 8)
    target_eras = np.equal(target, 7), 
    target_era_rings = np.equal(target, 11)
    
    target_bg = np.equal(target, 0)
    target_mid_region =  np.logical_or(
        np.logical_or(
            np.logical_or(target_hair, target_belowface),
            target_eras
        ),
        target_era_rings
    )
    
    # 找到 source 的脸部区域，也就是 除了背景、头发、belowface的区域
    source_hair_bg_neck_region = np.logical_or(
        np.logical_or(np.equal(source, 0), np.equal(source, 4)),
        np.equal(source, 8)
    )
    source_ear_earings_region = np.logical_or(np.equal(source, 7), np.equal(source, 11))
    source_non_face_region = np.logical_or(source_hair_bg_neck_region, source_ear_earings_region)
    source_face_region = np.logical_not(source_non_face_region)
    # source_face_region = np.logical_not(source_hair_bg_neck_region)
    
    # 先贴target的背景、belowface，耳朵 以及， 耳环
    res[target_bg] = 99  # a magic number, 先占住背景区域，用99标记
    res[target_belowface] = 8
    # # # 再贴target的头发
    # res[target_hair] = 4  ## 这一步放在贴source的前还是后，会影响是否保留target的刘海
    res[target_eras] = 7
    res[target_era_rings] = 11
    
    # 再贴source的脸部
    res[source_face_region] = source[source_face_region]
    
    # 再贴target的头发
    res[target_hair] = 4
    
    # 剩余是0的地方，我们填皮肤
    if np.sum(res==0) != 0:
        hole_map = 255*(res==0)
        res[res==0] = 6
    else:
        hole_map = np.zeros_like(res)
        
    # 最后把背景的label恢复一下
    res[res==99] = 0

    try:
        eye_line = np.where(res == 2)[0].min()
    except:
        eye_line = np.where(res == 3)[0].min()
     
    return res, hole_map, eye_line


def swap_head_mask_revisit_considerGlass(source, target):
    res = np.zeros_like(target)
    
    # 先找到target的 hair 区域 , belowface 区域, 耳朵区域
    target_regions = [np.equal(target, i) for i in range(12)]
    source_regions = [np.equal(source, i) for i in range(12)]
    
    
    # 先贴target的背景、belowface，耳朵 以及， 耳环
    res[target_regions[0]] = 99  # a magic number, 先占住背景区域，用99标记
    res[target_regions[8]] = 8 # neck
    # res[target_regions[7]] = 7
    # res[target_regions[11]] = 11
    
    # 再贴source的脸部

    res[np.logical_and(source_regions[7], np.not_equal(res,99))] = 7
    res[np.logical_and(source_regions[11], np.not_equal(res,99))] = 11 
    res[np.logical_and(source_regions[1], np.not_equal(res,99))] = 1 # lip
    res[np.logical_and(source_regions[2], np.not_equal(res,99))] = 2 # eyebrows
    res[np.logical_and(source_regions[3], np.not_equal(res,99))] = 3 # eyes
    res[np.logical_and(source_regions[5], np.not_equal(res,99))] = 5 # nose
    res[np.logical_and(source_regions[6], np.not_equal(res,99))] = 6  # skin
    res[np.logical_and(source_regions[9], np.not_equal(res,99))] = 9 # mouth
    
    """
    # res[source_regions[7]] = 7
    # res[source_regions[11]] = 11
    res[source_regions[1]] = 1 # lip
    res[source_regions[2]] = 2 # eyebrows
    res[source_regions[3]] = 3 # eyes
    res[source_regions[5]] = 5 # nose
    res[source_regions[6]] = 6  # skin
    res[source_regions[9]] = 9 # mouth
    """
    
    # 再贴target的头发
    res[target_regions[10]] = 10 # eye_glass
    res[source_regions[4]] = 4  # hair
    
    # 剩余是0的地方，我们填皮肤
    if np.sum(res==0) != 0:
        hole_map = 255 * (res==0)

        # ear_line = np.where(res == 7)[0].max()
        res[res==0] = 6
        # res[res==0] = 6
        # bottom_half = res[ear_line:, :]
        # up_half = res[:ear_line, :]
        # bottom_half[bottom_half == 0] = 8
        # up_half[up_half == 0] = 6
        # res[res == 0] == 6
        # res[ear_line:, :] = bottom_half
    else:
        hole_map = np.zeros_like(res)
        
    # 最后把背景的label恢复一下
    res[res==99] = 0

    eyebrows_line = np.where(res == 2)[0].min()
     
    return res, hole_map, eyebrows_line


def fill_hole(mask, hole_mask, radius, eye_line, nose_line):
    hole = hole_mask.copy()
    k = 0
    while k < 500:
        k += 1
        mask_new = mask.copy()
        n, m = len(mask), len(mask[0])
        for i in range(n):
            for j in range(m):
                if hole[i][j]:
                    """
                    if i < eye_line:
                        mask_new[i][j] = 6
                        hole[i][j] = False
                        continue
                    """
                    if i < nose_line:
                        neighbors = mask[max(0, i - radius): min(i + radius, n - 1), j-2: j+2]
                    else:
                        neighbors = mask[max(0, i - radius): min(i + radius, n - 1), j]
                    neighbors = list(neighbors[neighbors != 1])
                    try:
                        mask_new[i][j] = max(neighbors, key=neighbors.count)
                        hole[i][j] = False
                    except:
                        pass
        mask = mask_new

        if not hole.any():
            break

    mask[hole] = 6

    return mask



def swap_head_mask_hole_first(source, target):
    """ segmentation format:
        0 - background
        1 - lip
        2 - eyebrow
        3 - eyes
        4 - hair
        5 - nose
        6 - skin
        7 - ear
        8 - neck
        9 - tooth
        10 - eyeglass
        11 - earring
    """
    # calculate the hole map fist
    source_bg_mask = np.logical_or(source == 4, source == 0)  # hair, bg
    source_bg_mask = np.logical_or(source_bg_mask, source == 8)  # neck
    source_bg_mask = np.logical_or(source_bg_mask, source == 7)  # ear
    source_bg_mask = np.logical_or(source_bg_mask, source == 11)  # earring
    source_face_mask = np.logical_not(source_bg_mask)

    target_bg_mask = np.logical_or(target == 4, target == 0)  # hair, bg
    target_bg_mask = np.logical_or(target_bg_mask, target == 8)  # neck
    target_bg_mask = np.logical_or(target_bg_mask, target == 7)  # ear
    target_bg_mask = np.logical_or(target_bg_mask, target == 11)  # earring
    target_face_mask = np.logical_not(target_bg_mask)

    face_overlap_mask = np.logical_and(source_face_mask, target_face_mask)
    hole_mask = np.logical_xor(face_overlap_mask, target_face_mask)

    # swap mask
    res = np.zeros_like(target)
    
    target_regions = [np.equal(target, i) for i in range(12)]
    source_regions = [np.equal(source, i) for i in range(12)]

    # adjust or finetune the hole mask
    eye_line = int(2 / 5 * target.shape[0])
    nose_line = int(3 / 5 * target.shape[0])
    if np.any(source == 3):
        eye_line = np.where(source == 3)[0].max()  # eye lowest
    elif np.any(source == 2):
        eye_line = np.where(source == 2)[0].max()  # eye_brow lowest
    if np.any(source == 5):
        nose_line = np.where(source == 5)[0].max()  # nose lowest
    # hole_mask[np.logical_and(source_regions[4], target_regions[6])] = False  # source hair & target skin, not
    # hole_mask[np.logical_and(source_regions[4], target_regions[2])] = False  # source hair & target eyebrow, not
    # hole_mask[np.logical_and(source_regions[4], target_regions[3])] = False  # source hair & target eye, not
    if len(hole_mask) >= eye_line:
        hole_mask[:eye_line, :] = False  # higher than eyes set as False

    """ The background, neck, ear and earrings regions of target """
    res[target_regions[0]] = 99  # a place-holder magic number for bg (target-bg)
    res[target_regions[8]] = 8  # neck (target-bg)
    # res[target_regions[4]] = 4  # hair, hair first as background
    res[target_regions[7]] = 7  # ear (target-bg)
    res[target_regions[11]] = 11  # earring (target-bg)

    # fill in the hole

    # res = fill_hole(res, hole_mask, radius=5, eye_line=eye_line, nose_line=nose_line)
    # res[hole_mask] = 4
    # hole_mask[:eye_line, :] = False  # higher than eyes set as False

    """ The inner-face of source """
    ''' op1. cairong version '''
    # res[source_regions[7]] = 7
    # res[source_regions[11]] = 11
    res[source_regions[1]] = 1  # lip
    res[source_regions[2]] = 2  # eyebrows
    res[np.logical_and(source_regions[4], target_regions[2])] = 2  # source hair & target eyebrows
    res[source_regions[3]] = 3  # eyes
    res[source_regions[5]] = 5  # nose
    res[source_regions[6]] = 6  # skin
    res[source_regions[9]] = 9  # mouth
    ''' op2. zhian version '''
    # res[np.logical_and(source_regions[1], np.not_equal(res, 99))] = 1  # lip
    # res[np.logical_and(source_regions[2], np.not_equal(res, 99))] = 2  # eyebrows
    # res[np.logical_and(source_regions[3], np.not_equal(res, 99))] = 3  # eyes
    # res[np.logical_and(source_regions[5], np.not_equal(res, 99))] = 5  # nose
    # res[np.logical_and(source_regions[6], np.not_equal(res, 99))] = 6  # skin
    # res[np.logical_and(source_regions[9], np.not_equal(res, 99))] = 9  # mouth

    """ Fix target foreground like hat occlusions """
    # Additional foreground = (target_bg) && (source_skin higher than target_skin)
    H, W = target.shape
    target_skin_highest_by_width = np.ones(W, dtype=np.long) * H
    target_skin = np.zeros_like(target, dtype=target.dtype)
    target_skin[target_regions[6]] = 1
    target_skin = target_skin * (np.arange(H)[:, None])
    target_skin[target_skin == 0] = H
    target_skin_highest_by_width = target_skin.min(axis=0)  # (W,)
    target_bg_region = np.where(target == 0)
    target_bg_positions_h = target_bg_region[0]
    target_bg_positions_w = target_bg_region[1]
    target_foreground_h_positions = []
    target_foreground_w_positions = []
    for i in range(len(target_bg_positions_h)):
        h = target_bg_positions_h[i]
        w = target_bg_positions_w[i]
        if h <= target_skin_highest_by_width[w] != H:
            target_foreground_h_positions.append(h)
            target_foreground_w_positions.append(w)
    target_foreground_region = (np.array(target_foreground_h_positions),
                                np.array(target_foreground_w_positions))
    if len(target_foreground_h_positions) > 0:
        res[target_foreground_region] = 98  # additional foreground (target-foreground)

    # res[np.logical_and(source_regions[6], np.not_equal(res, 99))] = 6  # skin
    res[target_regions[4]] = 4  # not hair first (target-foreground), hair as foreground
    res[target_regions[10]] = 10  # eye_glass (target-foreground)
    # res[target_regions[7]] = 7  # removed, ear is background (target-background)

    """ The missing pixels, fill in skin temporarily """
    ''' op1. cairong version '''
    res[res == 0] = 6  # fill hole with skin
    res[res == 99] = 0
    res[res == 98] = 0
    hole_map = res.copy()
    hole_map[hole_mask] = 17  # see: torch_utils.get_colors
    ''' op2. zhian version '''
    # if np.sum(res == 0) != 0:
    #     hole_mask = 1 * (res == 0)
    #     res[res == 0] = 6  # skin
    # else:
    #     hole_mask = np.zeros_like(res)
    # hole_mask = hole_mask.astype(np.bool)
    # # hole_mask[0:eye_line] = False  # set parts higher than eyes to zero(False)
    # hole_mask[source_regions[4]] = False  # set source hair parts to zero(False)
    # res[res == 99] = 0  # restore the background
    # hole_map = res.copy()
    # hole_map[hole_mask] = 1

    """
    res: 0-bg, 1-lip, 2-eyebrow, 3-eye, 4-hair, 5-nose, 6-skin, 7-ear, 8-neck
    hole_mask: in {True,False}
    hole_map: in {0,...,11}
    """
    return res, hole_mask, hole_map, nose_line


def swap_comp_style_vector(style_vectors1, style_vectors2, comp_indices=[], belowFace_interpolation=False):
    """替换 style_vectors1 中某个component的 style vectors

    Args:
        style_vectors1 (Tensor): with shape [1,#comp,512], target image 的 style vectors
        style_vectors2 (Tensor): with shape [1,#comp,512], source image 的 style vectors
    """
    assert comp_indices is not None

    style_vectors = copy.deepcopy(style_vectors1)

    for comp_idx in comp_indices:
        style_vectors[:, comp_idx, :] = style_vectors2[:, comp_idx, :]

    # 额外处理一下耳朵 和 耳环部分

    # 如果 source 没有耳朵，那么就用 source 和target 耳朵style vectors的平均值 (为了和皮肤贴合)
    # if torch.sum(style_vectors2[:,7,:]) == 0:
    style_vectors[:, 7, :] = (style_vectors1[:, 7, :] + style_vectors2[:, 7, :]) / 2

    # 直接用 target 耳环的style vector
    style_vectors[:, 11, :] = style_vectors1[:, 11, :]

    # 脖子用二者的插值
    if belowFace_interpolation:
        style_vectors[:, 8, :] = (style_vectors1[:, 8, :] + style_vectors2[:, 8, :]) / 2

    # 如果source 没有牙齿，那么用 target 牙齿的 style vector
    if torch.sum(style_vectors2[:, 9, :]) == 0:
        style_vectors[:, 9, :] = style_vectors1[:, 9, :]

    return style_vectors


def swap_head_mask_target_bg_dilation(source, target):
    target_bg_regions = np.logical_or(target == 4, target == 0)
    target_bg_regions = np.logical_or(target_bg_regions, target == 8)
    target_bg_regions = np.logical_or(target_bg_regions, target == 7)
    target_bg_regions = np.logical_or(target_bg_regions, target == 11)
    
    target_bg_mask = np.ones_like(target)
    target_bg_mask[target_bg_regions] = target[target_bg_regions]
    target_bg_mask[target_bg_mask == 0] = 99
    target_bg_mask[target_bg_mask == 1] == 0
    target_bg_mask = torch.from_numpy(target_bg_mask).unsqueeze(0).unsqueeze(0).cuda().float()
    radius = 3
    target_bg_mask = dilation(target_bg_mask, 
                                torch.ones(2 * radius + 1, 2 * radius + 1, device=target_bg_mask.device), 
                                engine='convolution')
    target_bg_mask = dilation(target_bg_mask, 
                                torch.ones(2 * radius + 1, 2 * radius + 1, device=target_bg_mask.device), 
                                engine='convolution')
    target_bg_mask = dilation(target_bg_mask, 
                                torch.ones(2 * radius + 1, 2 * radius + 1, device=target_bg_mask.device), 
                                engine='convolution')
    target_bg_mask = dilation(target_bg_mask, 
                                torch.ones(2 * radius + 1, 2 * radius + 1, device=target_bg_mask.device), 
                                engine='convolution')
    target_bg_mask = dilation(target_bg_mask, 
                                torch.ones(2 * radius + 1, 2 * radius + 1, device=target_bg_mask.device), 
                                engine='convolution')
    target_bg_mask = dilation(target_bg_mask, 
                                torch.ones(2 * radius + 1, 2 * radius + 1, device=target_bg_mask.device), 
                                engine='convolution')
    target_bg_mask = dilation(target_bg_mask, 
                                torch.ones(2 * radius + 1, 2 * radius + 1, device=target_bg_mask.device), 
                                engine='convolution')
    


    res = target_bg_mask[0][0].cpu().detach().int().numpy()
    res[res == 99] = 0

    # swap mask
    # target_regions = [np.equal(target, i) for i in range(12)]
    # source_regions = [np.equal(source, i) for i in range(12)]
    
    # res[target_regions[0]] = 0  
    # res[target_regions[8]] = 8 # neck
    # res[target_regions[4]] = 4  # hair
    # res[target_regions[7]] = 7
    # res[target_regions[11]] = 11
    
    # res[source_regions[7]] = 7
    # res[source_regions[11]] = 11
    res[source == 1] = 1 # lip
    res[source == 2] = 2 # eyebrows
    res[source == 3] = 3 # eyes
    res[source == 5] = 5 # nose
    res[source == 6] = 6  # skin
    res[source == 9] = 9 # mouth
    
    res[target == 4] = 4
    res[target == 10] = 10 # eye_glass

    hole_map = np.zeros_like(res)

    try:
        eye_line = np.where(res == 3)[0].max()
    except:
        eye_line = np.where(res == 5)[0].min()
     
    return res, hole_map, hole_map, eye_line