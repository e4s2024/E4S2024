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
import importlib
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
from swap_face_fine.face_inpainting import inpainting
from utils.paste_back_tricks import Trick, SoftErosion
# from utils.swap_face_mask import swap_head_mask_revisit_considerGlass

from swap_face_fine.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps

from swap_face_fine.swap_face_mask import swap_head_mask_revisit, swap_head_mask_target_bg_dilation
from swap_face_fine.swap_face_mask import swap_head_mask_hole_first

from utils.alignment import crop_faces, calc_alignment_coefficients
from utils.morphology import dilation, erosion, opening
from utils.util import save, get_5_from_98, get_detector, get_lmk
from swap_face_fine.MISF.inpainting import inpainting_face
from swap_face_fine.deformation_demo import image_deformation

# from PIPNet.lib.tools import get_lmk_model, demo_image
from alignment import norm_crop, norm_crop_with_M
from swap_face_fine.Blender.inference import BlenderInfer


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_default(x, target_str, **kwargs):
    if x is None:
        return get_obj_from_str(target_str)(**kwargs)
    return x


def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)


def logical_and_reduce(*tensors):
    return torch.stack(tensors, dim=0).all(dim=0)


def crop_and_align_face(target_files):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False

    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, target_files, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    # quads: [(4,2),...]

    # crop的逆变换，用于后期贴回到原始视频上去
    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]

    return crops, orig_images, quads, inv_transforms


class FaceSwap(object):
    def __init__(self):
        # face parsing model
        faceParsing_ckpt = "./pretrained/faceseg/79999_iter.pth"
        self.faceParsing_model = init_faceParsing_pretrained_model(faceParsing_ckpt)

        opts = OurSwapFacePipelineOptions().parse()
        net = Net3(opts)
        net = net.to(opts.device)
        save_dict = torch.load(opts.checkpoint_path)
        net.load_state_dict(torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module."))
        net.latent_avg = save_dict['latent_avg'].to(opts.device)
        print("Load LocalStyleGAN pre-trained model success!")

        self.opts = opts
        self.net = net

        self.pose_est = PoseEstimator(weights='./pretrained/pose/model_weights.zip')

        self.GPEN_model = None
        self.face_enhancer = None
        self.face_enhancer_cf = None
        self.face_enhancer_esr = None
        self.blender = BlenderInfer()

        self.vid2_generator = None
        self.vid2_kp_detector = None
        self.vid2_he_estimator = None
        self.vid2_estimate_jacobian = None

        self.mask_softer = SoftErosion().to(opts.device)

    def get_cropped_source_target_result(self):
        return self.S, self.T, self.cropped_output

    def _blending_two_faces(self, swapped_face_image, enhancement_mode):
        swapped_face_mask = faceParsing_demo(self.faceParsing_model, swapped_face_image, convert_to_seg12=False)
        if self.verbose:
            Image.fromarray(swapped_face_mask).save(os.path.join(self.save_dir, "swapped_face_mask.png"))
        # self.blender = BlenderInfer() if self.blender is None else self.blender
        self.blender = get_default(self.blender,
                                   "swap_face_fine.Blender.inference.BlenderInfer")
        swapped_face_image = self.blender.infer_image(swapped_face_image, self.T,
                                                Image.fromarray(swapped_face_mask),
                                                Image.fromarray(self.T_mask_ori_seg))

        swapped_face_image = self._face_enhancement(swapped_face_image, enhancement_mode)
        
        return swapped_face_image

    @staticmethod
    def _nchw_torch_to_nhwc_np(x: torch.Tensor):
        from einops import rearrange
        x = rearrange(x, "n c h w -> n h w c").contiguous()
        x = x.cpu().numpy()
        x = x.squeeze(0)
        x = x.squeeze(-1)
        return x
            
    def _past_back(self, swapped_face_image, swapped_msk, crop_mode,
                   use_face_inpainting: bool = False,
                   verbose: bool = True,):
        # 最后贴回去
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
            10 -
            11 - earring
        """
        outer_dilation = 2  # 这个值可以调节
        mask_bg = logical_or_reduce(*[swapped_msk == clz for clz in [0, 11, 4, 7, 8]])   # 4,8,7  # 如果是视频换脸，考虑把头发也弄进来当做背景的一部分, 11 earrings 4 hair 8 neck 7 ear
        is_foreground = torch.logical_not(mask_bg)
        hole_index = self.hole_mask[None][None]
        is_foreground[hole_index[None]] = True
        foreground_mask = is_foreground.float()  # (1,1,512,512)
        debug = False

        if debug:
            vis_img = Image.fromarray((self._nchw_torch_to_nhwc_np(foreground_mask) * 255).astype(np.uint8))
            vis_img.save(os.path.join(self.save_dir, 'Step3a_foreground_mask.png'))
        
        # foreground_mask = dilation(foreground_mask, torch.ones(2 * outer_dilation + 1, 2 * outer_dilation + 1, device=foreground_mask.device), engine='convolution')
        content_mask, border_mask, full_mask = self._create_masks(foreground_mask, operation='expansion',
                                                                  radius=outer_dilation)

        if debug:
            vis_img = Image.fromarray((self._nchw_torch_to_nhwc_np(content_mask) * 255).astype(np.uint8))
            vis_img.save(os.path.join(self.save_dir, 'Step3a_content_mask.png'))
            vis_img = Image.fromarray((self._nchw_torch_to_nhwc_np(border_mask) * 255).astype(np.uint8))
            vis_img.save(os.path.join(self.save_dir, 'Step3a_border_mask.png'))
            vis_img = Image.fromarray((self._nchw_torch_to_nhwc_np(full_mask) * 255).astype(np.uint8))
            vis_img.save(os.path.join(self.save_dir, 'Step3a_full_mask.png'))

        # past back
        content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
        content_mask = content_mask[0, 0, :, :, None].cpu().numpy()
        border_mask = F.interpolate(border_mask, (1024, 1024), mode='bilinear', align_corners=False)
        border_mask = border_mask[0, 0, :, :, None].cpu().numpy()
        border_mask = np.repeat(border_mask, 3, axis=-1)

        # content_mask = F.interpolate(content_mask, (1024, 1024), mode='bilinear', align_corners=False)
        '''
        full_mask = F.interpolate(full_mask, (1024, 1024), mode='bilinear', align_corners=False)
        full_mask_image = Image.fromarray(255 * full_mask[0,0,:,:].cpu().numpy().astype(np.uint8)) 
        pasted_image = self._smooth_face_boundry(swapped_face_image, self.T, full_mask_image, radius=outer_dilation)
        pasted_image = pasted_image.convert('RGB')
        '''

        swapped_and_pasted = swapped_face_image * content_mask + self.T * (1 - content_mask)
        swapped_and_pasted = Image.fromarray(np.uint8(swapped_and_pasted))
        swapped_and_pasted = Image.fromarray(blending(np.array(self.T),
                                                      np.array(swapped_and_pasted), mask=border_mask))

        # face inpainting
        if use_face_inpainting:
            swapped_and_pasted_inpaint = inpainting(swapped_and_pasted, self.hole_mask)
            if verbose:
                swapped_and_pasted.save(os.path.join(self.save_dir, 'Step4a_inpainting_input.png'))
                # swapped_and_pasted_inpaint.save(os.path.join(self.save_dir, 'Step3b_inpainting_generated.png'))
            blending_mask = self.hole_mask.astype(np.float32).clip(0., 1.)
            if verbose:
                # blending_mask_img = Image.fromarray((blending_mask * 255).astype(np.uint8))
                # blending_mask_img.save(os.path.join(self.save_dir, 'Step3b_blending_mask.png'))
                # content_mask_img = Image.fromarray((content_mask.squeeze().astype(np.float32) * 255).astype(np.uint8))
                # content_mask_img.save(os.path.join(self.save_dir, 'Step3c_content_mask.png'))
                pass
            blending_mask = torch.FloatTensor(blending_mask).to(swapped_msk.device)
            blending_mask = blending_mask.unsqueeze(0).unsqueeze(0)
            blending_mask = F.interpolate(blending_mask, size=swapped_and_pasted.size,
                                          mode="bilinear", align_corners=True)
            blending_mask, _ = self.mask_softer(blending_mask)  # (1,1,1024,1024), in [0.,1.]
            blending_mask = blending_mask.clamp(0, 1).cpu().numpy()
            swapped_and_pasted = Trick.blending_two_images_with_mask(
                swapped_and_pasted,
                swapped_and_pasted_inpaint.resize(swapped_and_pasted.size,
                                                  resample=Image.BILINEAR),
                up_ratio=1.0, up_mask=blending_mask
            )

            if verbose:
                swapped_and_pasted.save(os.path.join(self.save_dir, 'Step4c_inpainting_blended.png'))

            blending_mask = torch.FloatTensor(content_mask).to(swapped_msk.device)
            blending_mask = blending_mask.squeeze()
            blending_mask = blending_mask.unsqueeze(0).unsqueeze(0)
            blending_mask = F.interpolate(blending_mask, size=swapped_and_pasted.size,
                                          mode="bilinear", align_corners=True)
            blending_mask, _ = self.mask_softer(blending_mask)  # (1,1,1024,1024), in [0.,1.]
            content_mask = blending_mask.clamp(0, 1).squeeze().unsqueeze(-1).cpu().numpy()  # (1024,1024,1)
            swapped_and_pasted = swapped_and_pasted * content_mask + self.T * (1 - content_mask)
            swapped_and_pasted = Image.fromarray(np.uint8(swapped_and_pasted))

        # self.swapped_and_pasted = swapped_and_pasted
        if verbose:
            swapped_and_pasted.save(os.path.join(self.save_dir, "Step3_swapped_and_pasted.png"))

        if crop_mode == 'target':                 
            inv_trans_coeffs, orig_image = self.inv_transforms[0], self.orig_images[0]
            swapped_and_pasted = swapped_and_pasted.convert('RGBA')
            pasted_image = orig_image.convert('RGBA')
            swapped_and_pasted.putalpha(255)
            projected = swapped_and_pasted.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
            pasted_image.alpha_composite(projected)

        elif crop_mode == 'both':                
            # 按照crop贴回去 
            inv_trans_coeffs, orig_image = self.inv_transforms[1], self.orig_images[1]
            swapped_and_pasted = swapped_and_pasted.convert('RGBA')
            pasted_image = orig_image.convert('RGBA')
            swapped_and_pasted.putalpha(255)
            projected = swapped_and_pasted.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
            pasted_image.alpha_composite(projected)

        else:
            '''
            # 直接贴，但是脸部和背景融合的地方 smooth一下
            if outer_dilation == 0:
                pasted_image = self._smooth_face_boundry(swapped_face_image, self.T, content_mask_image, radius=outer_dilation)
            else:
                pasted_image = self._smooth_face_boundry(swapped_face_image, self.T, full_mask_image, radius=outer_dilation)
            '''
            pasted_image = swapped_and_pasted

        return pasted_image

    def _inpaint_face(self, input_image, swapped_msk, verbose: bool = False):
        hole_mask = (self.hole_mask.clip(0., 1.) * 255).astype(np.uint8)
        hole_mask = cv2.dilate(hole_mask, kernel=(3, 3), iterations=1)  # 1st dilate
        hole_mask = cv2.GaussianBlur(hole_mask, ksize=(3, 3), sigmaX=11)
        hole_mask[hole_mask > 0] = 255  # hard mask
        hole_mask = (hole_mask.astype(np.float32) / 255.).clip(0., 1.)
        inpaint_image = inpainting(input_image, hole_mask)
        if verbose:
            input_image.save(os.path.join(self.save_dir, 'Step4a_inpainting_input.png'))
            inpaint_image.save(os.path.join(self.save_dir, '256x_Step4b_inpainting_generated.png'))
            inpaint_image.resize(input_image.size,
                                 resample=Image.BICUBIC).save(
                os.path.join(self.save_dir, '256to1024_Step4b_inpainting_generated.png'))
        blending_mask = (self.hole_mask.clip(0., 1.) * 255).astype(np.uint8)
        blending_mask = cv2.dilate(blending_mask, kernel=(5, 5), iterations=1)  # 1st dilate
        # blending_mask = cv2.erode(blending_mask, kernel=(2, 2), iterations=4)
        blending_mask = cv2.GaussianBlur(blending_mask, ksize=(7, 7), sigmaX=11)
        blending_mask[blending_mask > 0] = 255  # 2nd dilate
        blending_mask = cv2.GaussianBlur(blending_mask, ksize=(3, 3), sigmaX=11)
        blending_mask = (blending_mask.astype(np.float32) / 255.).clip(0., 1.)
        if verbose:
            hole_mask_img = Image.fromarray((self.hole_mask.clip(0., 1.).astype(np.uint8) * 255).astype(np.uint8))
            hole_mask_img.save(os.path.join(self.save_dir, 'mask_Step4b_hole_mask.png'))
            blending_mask_img = Image.fromarray((blending_mask * 255).astype(np.uint8))
            blending_mask_img.save(os.path.join(self.save_dir, 'mask_Step4b_soft_blending_mask.png'))
        blending_mask = torch.FloatTensor(blending_mask).to(swapped_msk.device)
        blending_mask = blending_mask.unsqueeze(0).unsqueeze(0)
        blending_mask = F.interpolate(blending_mask, size=input_image.size,
                                      mode="bilinear", align_corners=True)
        blending_mask, _ = self.mask_softer(blending_mask)  # (1,1,1024,1024), in [0.,1.]
        blending_mask = blending_mask.clamp(0., 1.).cpu().numpy()

        inpaint_image = self._face_enhancement(inpaint_image, mode="codeformer")
        inpaint_blend_image = Trick.blending_two_images_with_mask(
            input_image,
            inpaint_image.resize(input_image.size, resample=Image.BICUBIC),
            up_ratio=1.0, up_mask=blending_mask
        )

        if verbose:
            inpaint_image.save(os.path.join(self.save_dir, "Step4c_inpainting_enhance.png"))
            inpaint_blend_image.save(os.path.join(self.save_dir, 'Step4c_inpainting_blended.png'))

        return inpaint_blend_image

    def _smooth_face_boundry(self, image, dst_image, mask, radius=0, sigma=0.0):
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

    def _copy_and_paste(self, swapped_face_image, swapped_msk, driven_mask):
        mask_face = logical_or_reduce(*[swapped_msk == item for item in [1, 2, 3, 5, 6, 9]])
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
        radius = 5
        driven_face_area = erosion(driven_face_area.float(), torch.ones(2 * radius + 1, 2 * radius + 1, device=driven_face_area.device), engine='convolution')
        
        mouth_mask = torch.logical_and(mouth_mask, driven_face_area)
        mouth_mask = torch.logical_and(mouth_mask, mask_face).float()

        swapped_face_image, mouth_mask_image, seam_mask_image = self._mouth_transfer(source=self.D, target=swapped_face_image, mouth_mask=mouth_mask)
        if self.verbose:
            mouth_mask_image.save(os.path.join(self.save_dir, 'mouth_mask.png'))
            seam_mask_image.save(os.path.join(self.save_dir, 'mouth_seam_mask.png'))

        return swapped_face_image

    def _mouth_transfer(self, source, target, mouth_mask):
        mouth_mask = F.interpolate(mouth_mask, size=1024, mode='bilinear').clamp_(0, 1)
        mouth_mask = (mouth_mask > 0).float()
        _, seam_mask, _ = self._create_masks(mouth_mask, operation='expansion', radius=5)
        seam_mask = seam_mask.cpu().numpy()[0][0][:, :, None]
        seam_mask = np.repeat(seam_mask, 3, axis=-1)

        mouth_mask = mouth_mask.cpu().numpy()[0][0][:, :, None]
        mouth_mask = np.repeat(mouth_mask, 3, axis=-1)

        combined = np.array(source) * mouth_mask + np.array(target) * (1 - mouth_mask)
        combined = Image.fromarray(blending(np.array(source), combined, mask=seam_mask))
        return combined, Image.fromarray(np.uint8(mouth_mask * 255)), Image.fromarray(np.uint8(seam_mask * 255))
    
    def _fine_tuning_network(self):
        from options.optim_options import OptimOptions
        from optimization import Optimizer

        optim_opts = OptimOptions().parse()
        optim_opts.ds_frac = 1.0
        optim_opts.id_lambda = 0.1
        optim_opts.face_parsing_lambda = 0.1
        optim_opts.finetune_steps = 200

        optimizer = Optimizer(optim_opts, net=self.net)
        with torch.set_grad_enabled(True):
            self.net = optimizer.finetune_net_online(target=self.D, mask=self.D_mask, style_vectors=self.driven_style_vector)

        del optimizer

    def _swap_comp_style_vector(self, style_vectors1, style_vectors2, comp_indices=[], belowFace_interpolation=False):
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

    def _swap_mask_and_style_vector(self, ct_mode):
        swapped_msk, hole_mask, hole_map, eye_line = swap_head_mask_hole_first(self.D_mask, self.T_mask)
        _, hole_mask_ts, _, _ = swap_head_mask_hole_first(self.T_mask, self.D_mask)
        if self.verbose:
            cv2.imwrite(os.path.join(self.save_dir, "Mask_Swapped.png"), swapped_msk * 10)
            cv2.imwrite(os.path.join(self.save_dir, "Mask_Driven.png"), self.D_mask * 10)
            cv2.imwrite(os.path.join(self.save_dir, "Mask_Target.png"), self.T_mask * 10)
            swappped_one_hot = torch_utils.labelMap2OneHot(torch.from_numpy(swapped_msk).unsqueeze(0).unsqueeze(0).long(), num_cls=12)
            torch_utils.tensor2map(swappped_one_hot[0]).save(os.path.join(self.save_dir, "swappedMaskVis.png"))

            hole_one_hot = torch_utils.labelMap2OneHot(torch.from_numpy(hole_map).unsqueeze(0).unsqueeze(0).long(), num_cls=19)
            torch_utils.tensor2map(hole_one_hot[0]).save(os.path.join(self.save_dir, "hole_map.png"))
        
        # 保留 trgt_style_vectors 的background, hair, era_rings, eye_glass 其余的全用 driven_style_vectors
        if ct_mode:
            comp_indices = set(range(self.opts.num_seg_cls)) - {0, 10, 4, 8, 7, 11}  # 10 glass, 8 neck
        else:
            comp_indices = set(range(self.opts.num_seg_cls)) - {0, 10, 4, 11}
        swapped_style_vectors = self._swap_comp_style_vector(self.target_style_vector, self.driven_style_vector, list(comp_indices), belowFace_interpolation=False)
        if self.verbose:
            torch.save(swapped_style_vectors, os.path.join(self.save_dir, "swapped_style_vec.pt"))
        
        self.swapped_msk = swapped_msk
        self.swapped_style_vectors = swapped_style_vectors
        self.hole_mask = hole_mask
        self.eye_line = eye_line
        self.hole_mask_st_ts = hole_mask + hole_mask_ts

    def _calculate_style_vectors(self, driven, driven_onehot, target, target_onehot, optimize_W):
        if optimize_W:
            from options.optim_options import OptimOptions
            from optimization import Optimizer

            optim_opts = OptimOptions().parse()
            optim_opts.ds_frac = 1.0
            optim_opts.id_lambda = 0.1
            optim_opts.face_parsing_lambda = 0.1
            optim_opts.W_steps = 200

            optimizer = Optimizer(optim_opts, net=self.net)

            with torch.set_grad_enabled(True):
                driven_style_vector = optimizer.optim_W_online(self.D, self.D_mask)
                target_style_vector = optimizer.optim_W_online(self.T, self.T_mask)

            del optimizer
        else:
            with torch.no_grad():
                driven_style_vector, _ = self.net.get_style_vectors(driven, driven_onehot) 
                target_style_vector, _ = self.net.get_style_vectors(target, target_onehot)
        
        if self.verbose:
            torch.save(driven_style_vector, os.path.join(self.save_dir,"D_style_vec.pt"))
            driven_style_codes = self.net.cal_style_codes(driven_style_vector)
            driven_face, _ , structure_feats = self.net.gen_img(torch.zeros(1,512,32,32).to(self.opts.device), driven_style_codes, driven_onehot)                
            driven_face_image = torch_utils.tensor2im(driven_face[0])
            driven_face_image.save(os.path.join(self.save_dir,"D_recon.png"))

            torch.save(target_style_vector, os.path.join(self.save_dir, "T_style_vec.pt"))
            target_style_codes = self.net.cal_style_codes(target_style_vector)
            target_face, _ , structure_feats = self.net.gen_img(torch.zeros(1,512,32,32).to(self.opts.device), target_style_codes, target_onehot)                
            target_face_image = torch_utils.tensor2im(target_face[0])
            target_face_image.save(os.path.join(self.save_dir, "T_recon.png"))

        self.driven_style_vector, self.target_style_vector = driven_style_vector, target_style_vector

    @torch.no_grad()
    def _color_transfer(self, img_a, img_t, ct_mode):
        print(f'color transfer mode: {ct_mode}')
        if ct_mode == 'blender':
            mask_a = faceParsing_demo(self.faceParsing_model, img_a, convert_to_seg12=False)
            self.blender = get_default(self.blender,
                                       "swap_face_fine.Blender.inference.BlenderInfer")
            face_blending = self.blender.infer_image(img_a, img_t,
                                                     Image.fromarray(mask_a),
                                                     Image.fromarray(self.T_mask_ori_seg))
            face_blending.save(os.path.join(self.save_dir, "256x_Step2_color_transfer_gen.png"))
            face_blending = face_blending.resize(img_a.size)
            face_blending = self._face_enhancement(face_blending, mode="realesr")
            face_blending.save(os.path.join(self.save_dir, "1024x_Step2_color_transfer_enhance.png"))
            
            return face_blending
        else:
            D, T = img_a, img_t
            driven_mask, target_mask = self.swapped_msk, self.target_mask
            mask_face_driven = logical_or_reduce(*[driven_mask == item for item in [1, 2, 3, 5, 6, 9, 7, 8]]).float()
            mask_face_target = logical_or_reduce(*[target_mask == item for item in [1, 2, 3, 5, 6, 9, 7, 8]]).float()

            # radius = 10
            # mask_face_swapped = dilation(mask_face_swapped, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask_face_swapped.device), engine='convolution')
            # mask_face_target = dilation(mask_face_target, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask_face_target.device), engine='convolution')
            mask_face_driven = F.interpolate(mask_face_driven, (1024, 1024), mode='bilinear', align_corners=False)
            mask_face_target = F.interpolate(mask_face_target, (1024, 1024), mode='bilinear', align_corners=False)

            _, face_border, _ = self._create_masks(mask_face_driven, operation='expansion', radius=10)
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
                                                    ct_mode=ct_mode)))
            
            # swapped_face_original = np.array(D)
            driven_face_image = D * (1 - mask_face_driven) + driven_face_inner * mask_face_driven
            # driven_face_image = Image.fromarray(np.uint8(driven_face_image))
            driven_face_image = Image.fromarray(blending(np.array(D), np.array(driven_face_image), mask=face_border))
            
            return driven_face_image

    def _warp_target(self):
        if self.crop_mode == 'target':
            inv_trans_coeffs, orig_image = self.inv_transforms[0], self.orig_images[0]

            source_trans = self.D.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
            target_warped = image_deformation(image=orig_image, image_ref=source_trans)
            crops, orig_images, quads, inv_transforms = crop_and_align_face([(self.source_and_target_files[1][0], target_warped)])
            crops = [crop.convert("RGB") for crop in crops]
            T = crops[0]
        elif self.crop_mode == 'both':
            inv_trans_coeffs, orig_image = self.inv_transforms[1], self.orig_images[1]
            
            source_trans = self.D.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
            target_warped = image_deformation(image=orig_image, image_ref=source_trans)
            crops, orig_images, quads, inv_transforms = crop_and_align_face([(self.source_and_target_files[0][0], source_trans), (self.source_and_target_files[1][0], target_warped)])
            crops = [crop.convert("RGB") for crop in crops]
            _, T = crops
        else:
            return        

        T_mask = faceParsing_demo(self.faceParsing_model, T, convert_to_seg12=True)
        T_mask_ori_seg = faceParsing_demo(self.faceParsing_model, T, convert_to_seg12=False)
        if self.verbose:
            Image.fromarray(T_mask).save(os.path.join(self.save_dir, "T_mask_deformed.png"))
            Image.fromarray(T_mask_ori_seg).save(os.path.join(self.save_dir, "T_mask_ori_seg.png"))
            T.save(os.path.join(self.save_dir, "T_cropped_deformed.png"))

        self.T = T
        self.T_mask = T_mask
        self.inv_transforms, self.orig_images = inv_transforms, orig_images
        self.T_mask_ori_seg = T_mask_ori_seg

    def _face_enhancement(self, face_img, mode: str = None):
        if mode is None:
            mode = self.enhancement_mode
        driven = np.array(face_img)
        if mode == 'gfpgan':
            from swap_face_fine.GFPGAN.inference_gfpgan import face_restoration
            driven = face_restoration([driven])
            res = Image.fromarray(driven[0])
        elif mode == 'gpen':
            from swap_face_fine.gpen.gpen_demo import init_gpen_pretrained_model, GPEN_demo

            self.GPEN_model = get_default(self.GPEN_model,
                                          "swap_face_fine.gpen.gpen_demo.init_gpen_pretrained_model")

            driven = GPEN_demo(driven[:, :, ::-1], self.GPEN_model, aligned=False)
            res = Image.fromarray(driven[:,:,::-1])
        elif mode == 'codeformer':
            from swap_face_fine.inference_codeformer import CodeFormerInfer
            self.face_enhancer_cf = get_default(self.face_enhancer_cf,
                                                "swap_face_fine.inference_codeformer.CodeFormerInfer")
            res = self.face_enhancer_cf.infer_image(face_img)
        elif mode == "SwinIR":
            from swap_face_fine.SwinIR.image_infer import SwinIRInfer
            self.face_enhancer = get_default(self.face_enhancer,
                                             "swap_face_fine.SwinIR.image_infer.SwinIRInfer",
                                             device=self.opts.device)
            res = self.face_enhancer.infer(face_img)
        elif mode == "realesr":
            from swap_face_fine.realesr.image_infer import RealESRBatchInfer
            self.face_enhancer_esr = get_default(
                self.face_enhancer_esr,
                "swap_face_fine.realesr.image_infer.RealESRBatchInfer",
            )
            res = self.face_enhancer_esr.infer_image(face_img)
        else:
            raise ValueError(f'Wrong face enhancement mode {face_enhancement}.')
        
        return res

    def _face_alignment(self, source, target, target_mask):
        if self.crop_mode == 'target':
            crops, orig_images, quads, inv_transforms = crop_and_align_face(self.source_and_target_files[1:])
            crops = [crop.convert("RGB") for crop in crops]
            T = crops[0]
            S = Image.open(source).convert("RGB").resize((1024, 1024))
        elif self.crop_mode == 'both':
            crops, orig_images, quads, inv_transforms = crop_and_align_face(self.source_and_target_files)
            crops = [crop.convert("RGB") for crop in crops]
            S, T = crops
        else:
            # if isinstance(source, Image):
            S = source.convert("RGB").resize((1024, 1024))
            # else:
            #     S = Image.open(source).convert("RGB").resize((1024, 1024))
            # if isinstance(target, Image):
            T = target.convert("RGB").resize((1024, 1024))
            # else:
            #     T = Image.open(target).convert("RGB").resize((1024, 1024))
            crops = [S, T]

        S_256, T_256 = [resize(np.array(im)/255.0, (256, 256)) for im in [S, T]]  # 256,[0,1]范围
        T_mask = faceParsing_demo(self.faceParsing_model, T, convert_to_seg12=True) if target_mask is None else target_mask

        T_mask_ori_seg = faceParsing_demo(self.faceParsing_model, T, convert_to_seg12=False)

        # self.S = self._face_enhancement(S)
        # self.T = self._face_enhancement(T)
        self.S = S
        self.T = T
        self.S_256 = S_256
        self.T_256 = T_256
        self.T_mask = T_mask
        if self.crop_mode == 'target' or self.crop_mode == 'both':
            self.orig_images = orig_images
            self.inv_transforms = inv_transforms
        self.T_mask_ori_seg = T_mask_ori_seg

        if self.verbose:
            Image.fromarray(T_mask).save(os.path.join(self.save_dir, "T_mask.png"))
            self.S.save(os.path.join(self.save_dir, "S_cropped.png"))
            self.T.save(os.path.join(self.save_dir, "T_cropped.png"))

    def _pose_alignment(self, pose_drive, pose_estimation, pose_threshold=15.):
        need_drive = False
        if pose_estimation:
            try:
                pose_s = self.pose_est.pose_from_image(image=np.array(S))
                pose_t = self.pose_est.pose_from_image(image=np.array(T))
                diff = (pose_s[0] - pose_t[0]) ** 2 + (pose_s[1] - pose_t[1]) ** 2 + (pose_s[2] - pose_t[2]) ** 2
                if math.sqrt(diff) > pose_threshold:
                    need_drive = True
            except:
                need_drive = True
        else:
            need_drive = True
        
        # need_drive = False
        S_256, T_256 = self.S_256, self.T_256
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

                face_vid2vid_cfg = "./pretrained/faceVid2Vid/vox-256.yaml"
                face_vid2vid_ckpt = "./pretrained/faceVid2Vid/00000189-checkpoint.pth.tar"

                if self.vid2_generator is None or self.vid2_kp_detector is None\
                        or self.vid2_he_estimator is None or self.vid2_estimate_jacobian is None:
                    self.vid2_generator, \
                    self.vid2_kp_detector, \
                    self.vid2_he_estimator, \
                    self.vid2_estimate_jacobian = init_facevid2vid_pretrained_model(
                        face_vid2vid_cfg, face_vid2vid_ckpt)

                predictions = drive_source_demo(S_256, [T_256],
                                                self.vid2_generator,
                                                self.vid2_kp_detector,
                                                self.vid2_he_estimator,
                                                self.vid2_estimate_jacobian)
                predictions = [(pred*255).astype(np.uint8) for pred in predictions]
            
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

                del generator, kp_detector, depth_encoder, depth_decoder
            
                driven = predictions[0]
            elif pose_drive == 'LIA':
                from swap_face_fine.LIA.run_demo import drive_demo

                driven = drive_demo(S_256, T_256)
                driven = (driven * 255).astype(np.uint8)
            else:
                raise ValueError(f'Wrong pose drive mode {pose_drive}.')
        else:
            driven = (S_256 * 255).astype(np.uint8)

        self.driven = Image.fromarray(driven)

    def _create_masks(self, mask, operation='dilation', radius=0):
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
            full_mask, _ = self.mask_softer(full_mask)
            erosion_mask, _ = self.mask_softer(erosion_mask)
            border_mask = full_mask - erosion_mask

        border_mask = border_mask.clip(0, 1)
        content_mask, _ = self.mask_softer(mask)
        
        return content_mask, border_mask, full_mask    

    def face_swap_pipeline(self,
                           source,
                           target,
                           save_dir,
                           target_mask=None,
                           crop_mode='both',
                           verbose=True,
                           optimize_W=False,
                           finetune_net=False,
                           copy_face=False,
                           pose_drive='faceVid2Vid',
                           pose_estimation=True,
                           enhancement_mode='gfpgan',
                           ct_mode='blender',
                           warp_target=True,
                           face_inpainting=True,
                           index: int = 0,
                           ):

        if not os.path.exists(save_dir) and verbose:
            os.makedirs(save_dir)

        self.save_dir = save_dir
        self.verbose = verbose
        self.face_inpainting = face_inpainting
        source_and_target_files = [source, target]
        if isinstance(source, str) and isinstance(target, str):
            self.source_and_target_files = [(os.path.basename(f).split('.')[0], f) for f in source_and_target_files]
        else:
            self.source_and_target_files = [('source', source), ('target', target)]
        self.result_name = "pasted_StepFinal_swap_%s_to_%s.png" % (self.source_and_target_files[0][0],
                                                                   self.source_and_target_files[1][0])
        self.crop_mode = crop_mode
        self.enhancement_mode = enhancement_mode

        # face alignment, get self.S and self.T
        self._face_alignment(source, target, target_mask)
        # pose alignment, get self.driven
        self._pose_alignment(pose_drive, pose_estimation)
        # face enhancement
        self.D = self._face_enhancement(self.driven, mode="gpen")  # fixed as GPEN

        # D mask
        D_mask = faceParsing_demo(self.faceParsing_model, self.D, convert_to_seg12=True)
        if self.verbose:
            Image.fromarray(D_mask).save(os.path.join(self.save_dir,"D_mask.png"))
            self.D.save(os.path.join(self.save_dir,"D_%s_to_%s.png"%(self.source_and_target_files[0][0],
                                                                     self.source_and_target_files[1][0])))
        self.D_mask = D_mask

        # warp the target face shape to the source, not used
        if warp_target:
            self._warp_target()

        # wrap input data
        driven_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(self.D_mask))
        driven_mask = (driven_mask * 255).long().to(self.opts.device).unsqueeze(0)
        driven_onehot = torch_utils.labelMap2OneHot(driven_mask, num_cls=self.opts.num_seg_cls)
        target_mask = transforms.Compose([TO_TENSOR])(Image.fromarray(self.T_mask))
        target_mask = (target_mask * 255).long().to(self.opts.device).unsqueeze(0)
        target_onehot = torch_utils.labelMap2OneHot(target_mask,
                                                    num_cls=self.opts.num_seg_cls)  # (1,12,512,512), in {0,1}

        self.driven_mask = driven_mask
        self.target_mask = target_mask

        driven = transforms.Compose([TO_TENSOR, NORMALIZE])(self.D)
        driven = driven.to(self.opts.device).float().unsqueeze(0)
        target = transforms.Compose([TO_TENSOR, NORMALIZE])(self.T)
        target = target.to(self.opts.device).float().unsqueeze(0)

        if self.verbose:
            torch_utils.tensor2map(driven_onehot[0]).save(os.path.join(self.save_dir, "D_mask_vis.png")) 
            torch_utils.tensor2map(target_onehot[0]).save(os.path.join(self.save_dir, "T_mask_vis.png")) 

        # calculate style vectors
        self._calculate_style_vectors(driven, driven_onehot, target, target_onehot, optimize_W)

        # recombination of mask and style vector
        self._swap_mask_and_style_vector(ct_mode)

        if finetune_net:
            self._fine_tuning_network()

        swapped_msk = Image.fromarray(self.swapped_msk).convert('L')
        swapped_msk = transforms.Compose([TO_TENSOR])(swapped_msk)
        swapped_msk = (swapped_msk * 255).long().to(self.opts.device).unsqueeze(0)
        rectangle_msk = torch.ones_like(swapped_msk, device=swapped_msk.device).long() * 6  # 6 means skin
        # swapped_msk = opening(swapped_msk, torch.ones(2 * 2 + 1, 2 * 2 + 1, device=swapped_msk.device), engine='convolution').long()
        swapped_onehot = torch_utils.labelMap2OneHot(swapped_msk, num_cls=self.opts.num_seg_cls)
        if self.verbose:
            torch_utils.tensor2map(swapped_onehot[0]).save(os.path.join(self.save_dir, "Seg_swapped_msk_vis.png"))
        # 换脸的结果            
        swapped_style_codes = self.net.cal_style_codes(self.swapped_style_vectors)
        swapped_face, _, structure_feats = self.net.gen_img(torch.zeros(1, 512, 32, 32).to(self.opts.device),
                                                            swapped_style_codes, swapped_onehot)
        swapped_face_image = torch_utils.tensor2im(swapped_face[0])

        self.swapped_msk = swapped_msk  # (1,1,512,512) in [0,...,9]

        if verbose:
            swapped_face_image.save(os.path.join(self.save_dir, 'Step1_gen_swapped_face.png'))

        if verbose:
            swapped_face_image_vis = self._past_back(swapped_face_image, swapped_msk, crop_mode,
                                                     use_face_inpainting=False, verbose=False)
            # swapped_face_image_vis = self._face_enhancement(swapped_face_image_vis, mode="gpen")
            swapped_face_image_vis.save(os.path.join(self.save_dir, 'pasted_Step1_gen_swapped_face.png'))

        # copy & paste mouth (not used)
        if copy_face:
            swapped_face_image = self._copy_and_paste(swapped_face_image, swapped_msk, driven_mask)

        # 2. color transfer
        if ct_mode:
            color_transfer_image = self._color_transfer(swapped_face_image, self.T, ct_mode)
            blending_mask = Trick.get_facial_mask_from_seg19(
                self.swapped_msk,
                target_size=color_transfer_image.size,
                edge_softer=self.mask_softer
            )  # 0.:blank, 1.:used pixels
            edge_img = Trick.get_edge(swapped_face_image)
            edge_img = np.array(edge_img).astype(np.float32) / 255.
            blending_mask = (blending_mask - edge_img).clip(0., 1.)  # remove high-frequency parts
            # blending_mask[blending_mask >= 0.5] = 1.
            swapped_face_image = Trick.blending_two_images_with_mask(
                swapped_face_image, color_transfer_image,
                up_ratio=0.75, up_mask=blending_mask.copy()
            )
            # swapped_face_image = Trick.sharp_image(swapped_face_image)
        if self.verbose and ct_mode:
            blending_mask_image = Image.fromarray((blending_mask.squeeze() * 255.).astype(np.uint8))
            blending_mask_image.save(os.path.join(self.save_dir, 'Step2_color_transfer_blending_mask.png'))
            blending_mask_image = Image.fromarray(((1 - blending_mask).squeeze() * 255.).astype(np.uint8))
            blending_mask_image.save(os.path.join(self.save_dir, 'Step2_color_transfer_blending_mask_inverse.png'))
            swapped_face_image.save(os.path.join(self.save_dir, 'Step2_color_transfer_blended.png'))
            swapped_face_image_vis = self._past_back(swapped_face_image, swapped_msk, crop_mode,
                                                     use_face_inpainting=False, verbose=False)
            # swapped_face_image_vis = self._face_enhancement(swapped_face_image_vis, mode="gpen")
            swapped_face_image_vis.save(os.path.join(self.save_dir, 'pasted_Step2_color_transfer.png'))

        # 3. paste_back to cropped target
        swapped_and_pasted = self._past_back(swapped_face_image, swapped_msk, crop_mode="",
                                             use_face_inpainting=False, verbose=self.verbose)
        if self.verbose:  # paste_back to ori full target
            swapped_and_pasted_vis = self._past_back(swapped_and_pasted, rectangle_msk, crop_mode,
                                                     use_face_inpainting=False, verbose=self.verbose)
            swapped_and_pasted_vis.save(os.path.join(self.save_dir, 'pasted_Step3_paste_back.png'))

        # 4. inpainting and paste back to the original target image
        # target_mask: (1,1,512,512), in {0,1,2,...,8}
        # TODO: first paste_back or inpaint? if inpaint last, cannot support full image; if paste last, double check.
        if self.face_inpainting:
            swapped_and_pasted = self._inpaint_face(swapped_and_pasted, swapped_msk, verbose=True)
            if self.verbose:  # paste_back to ori full target
                swapped_face_image_vis = self._past_back(swapped_and_pasted, rectangle_msk, crop_mode,
                                                         use_face_inpainting=False, verbose=False)
                # swapped_face_image_vis = self._face_enhancement(swapped_and_pasted, mode="codeformer")
                # swapped_face_image_vis = swapped_and_pasted
                swapped_face_image_vis.save(os.path.join(self.save_dir, 'pasted_Step4_inpaint.png'))

        self.cropped_output = swapped_and_pasted
        self.swapped_and_pasted = self._past_back(swapped_and_pasted, rectangle_msk, crop_mode,
                                                  use_face_inpainting=False, verbose=False)
        if verbose:
            self.swapped_and_pasted.save(os.path.join(self.save_dir, self.result_name))

        return self.swapped_and_pasted


if __name__ == '__main__':

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    img_root = '../datasets/CelebAMask-HQ/CelebA-HQ-img'
    img_paths = glob.glob(img_root + '/*.jpg')
    print(f'{img_root} total files:', len(img_paths))

    save_root = "./outputs/"
    os.system(f'rm -r {save_root}')

    pose_drive = 'faceVid2Vid'
    face_enhancement = 'gpen'  # 'gpen', 'codeformer', "SwinIR"

    face_swap = FaceSwap()

    from datasets.dataset import CelebAHQDataset
    from torch.utils.data import DataLoader

    use_self_source_target = False
    celeba_hq_folder = "/home/yuange/datasets/CelebA-HQ/"
    specific_mode_imgs = [
        ("test", "28063.jpg"),
        ("test", "28426.jpg"),
        ("test", "28297.jpg"),
        ("test", "28629.jpg"),
        ("train", "4829.jpg"),
        ("train", "5982.jpg"),
        ("train", "4612.jpg"),
        ("train", "4811.jpg"),
        ("test", "29404.jpg"),
        ("test", "29386.jpg"),
        ("test", "28740.jpg"),
        ("test", "28393.jpg"),
        ("test", "28072.jpg"),
        ("test", "29318.jpg"),
        ("test", "29989.jpg"),
        ("test", "28835.jpg"),
        ("train", "724.jpg"),
        ("train", "1123.jpg"),
        ("test", "29756.jpg"),
        ("test", "29220.jpg"),
        ("test", "28021.jpg"),
        ("test", "29833.jpg")
    ]

    ''' op1. CVPR dataset '''
    # dataset_celebahq = CelebAHQDataset(
    #     celeba_hq_folder,
    #     mode="all",
    #     specific_ids=specific_mode_imgs,
    #     load_vis_img=True,
    #     paired=True
    # )
    ''' op2. free dataset '''
    dataset_celebahq = CelebAHQDataset(
        celeba_hq_folder,
        mode="test",
        specific_ids=None,
        load_vis_img=True,
        paired=True,
        shuffle=True
    )
    ''' op3. single source and target file '''
    # use_self_source_target = True

    dataloader = DataLoader(
        dataset_celebahq,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )
    for idx, batch in enumerate(tqdm(dataloader)):
        source_bag = batch["bag1"]
        target_bag = batch["bag2"]

        sources = source_bag[3]  # path
        targets = target_bag[3]  # path
        source = sources[0]
        target = targets[0]
        # print(source, target)
        # continue

        torch.cuda.empty_cache()

        source_name = os.path.basename(source).split('.')[0]
        target_name = os.path.basename(target).split('.')[0]
        if source_name == target_name:
            continue

        idx_str = str("%05d" % idx)
        save_name = f"{idx_str}_{source_name}_to_{target_name}"
        save_dir = os.path.join(save_root, save_name)  # + '_no_blender'

        crop_mode = ""
        if use_self_source_target:
            source = "/home/yuange/Documents/E4S_v2/figs/method01/image45.jpeg"
            target = "/home/yuange/Documents/E4S_v2/figs/method01/image42.jpeg"
            save_dir = "/home/yuange/Documents/E4S_v2/figs/method01/outputs/"
            crop_mode = "both"

        swap_result = face_swap.face_swap_pipeline(source=Image.open(source),
                                                   target=Image.open(target),
                                                   save_dir=save_dir,
                                                   target_mask=None,
                                                   crop_mode=crop_mode,
                                                   verbose=True,
                                                   optimize_W=False,
                                                   finetune_net=False,
                                                   copy_face=False,  # copy mouth
                                                   pose_drive=pose_drive,
                                                   pose_estimation=False,  # 姿态相差＞某个阈值才做姿态对齐，也就是人脸重演
                                                   enhancement_mode=face_enhancement,  # face enhancing
                                                   ct_mode='blender',  # default: 'blender'
                                                   warp_target=False,  # not used
                                                   face_inpainting=True)

        if not use_self_source_target:
            swap_result.save(os.path.join(
                "/home/yuange/Documents/E4S_v2/sota_method_results/e4s_v2",
                f"{idx_str}.jpg"
            ))

        gpu_free, gpu_total = torch.cuda.mem_get_info()
        print(f'finished {save_name}. '
              f'GPU Free = {gpu_free / 1024 / 1024 / 1024} GB, '
              f'Total = {gpu_total / 1024 / 1024 / 1024} GB.')

        if idx >= 200:
            exit()
            continue

        if use_self_source_target:  # only infer once
            exit()
