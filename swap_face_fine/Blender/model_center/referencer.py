import torch
from torchvision import models
import torch.nn as nn
from .backbone import AdaptiveFeatureGenerator, SmallFPN
import torch.nn.functional as F
from .semantic_tools import get_part_dict, get_greyscale_head, get_dilated_mask, get_color_refer
import numpy as np


class Referencer(nn.Module):
    def __init__(self, args):
        super(Referencer, self).__init__()
        self.args = args
        if args.small_FPN:
            self.FPN = SmallFPN()
        else:
            self.FPN = AdaptiveFeatureGenerator(args)
        self.trainable_tao = nn.Parameter(torch.tensor(1.))

        if args.lambda_CYC != 0:
            self.compute_inv = True
        return

    def forward(self, img_A, img_T, mask_A, mask_T):
        """
        img_A/img_T: (1,3,256,256)
        mask_A/mask_T: (1,256,256)
        """
        feats_A = self.FPN(img_A, img_A)  # (1,256,64,64)

        # Random flip
        if np.random.rand() < 0.5:
            feats_T = self.FPN(img_T, img_T)  # (1,256,64,64)
        else:
            feats_T = self.FPN(torch.flip(img_T, dims=[-1]), torch.flip(img_T, dims=[-1]))
            # feats_T = torch.flip(feats_T, dims=[-1])

        # Calculate mask parts and extract image pixels
        part_dict_A = get_part_dict(mask_A)  # {'head':(1,256,256);...}
        part_dict_T = get_part_dict(mask_T)

        grayscale_head_A = get_greyscale_head(img_A, part_dict_A['head'])  # (1,256,256)
        inpainting_mask_T = (get_dilated_mask(part_dict_T['head']) - part_dict_T['head']).clamp(0, 1)
        e_AT = get_dilated_mask((part_dict_A['head'] + part_dict_T['head']).clamp(0, 1))
        inpainting_mask_A = (e_AT - part_dict_A['head']).clamp(0, 1)
        img_bg = img_T * (1 - e_AT[:, None])

        part_dict_A['inpainting'] = inpainting_mask_A
        part_dict_T['inpainting'] = inpainting_mask_T

        color_refs, color_inv_refs_pair = get_color_refer(img_T, feats_A, feats_T, part_dict_A, part_dict_T,
                                                          self.trainable_tao, self.compute_inv, self.args.small_FPN)

        if self.args.lambda_CYC2 != 0:
            # color_refs_cro, color_inv_refs_pair_cro = get_color_refer(
            #     torch.flip(img_T, dims=[0]),
            #     torch.flip(feats_A, dims=[0]),
            #     torch.flip(feats_T, dims=[0]),
            #     {k: torch.flip(v, dims=[0]) for k, v in part_dict_A.items()},
            #     {k: torch.flip(v, dims=[0]) for k, v in part_dict_T.items()},
            #     self.trainable_tao, self.compute_inv,
            #     self.args.small_FPN)

            color_refs_cro, color_inv_refs_pair_cro = get_color_refer(
                torch.flip(img_T, dims=[0]),
                feats_A,
                torch.flip(feats_T, dims=[0]),
                part_dict_A,
                {k: torch.flip(v, dims=[0]) for k, v in part_dict_T.items()},
                self.trainable_tao, self.compute_inv,
                self.args.small_FPN)

        if len(color_refs) <= 1:
            head_ref = torch.zeros_like(img_T)
            inpaint_ref = torch.zeros_like(img_T)
        else:
            head_ref = sum([v for k, v in color_refs.items() if k != 'inpainting'])
            inpaint_ref = color_refs['inpainting']

        packages = torch.cat([F.upsample_bilinear(torch.cat([head_ref,
                                                             inpaint_ref], dim=1), img_T.size()[-2:]),
                              part_dict_A['head'][:, None].float(),
                              inpainting_mask_A[:, None].float(),
                              grayscale_head_A[:, None],
                              img_bg], dim=1)
        return packages, color_inv_refs_pair, color_inv_refs_pair_cro
