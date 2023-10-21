import torch
import torch.nn as nn
from .referencer import Referencer
from .res_u_net import ResUNet


class Blender(nn.Module):
    def __init__(self, args):
        super(Blender, self).__init__()
        self.referencer = Referencer(args)
        self.unet = ResUNet(args)

    def forward(self, img_A, img_T, mask_A, mask_T):
        """
        img_A/img_T: (1,3,256,256)
        mask_A/mask_T: (1,256,256)
        """
        packages, color_inv_refs_pair, color_inv_refs_pair_cro = self.referencer(img_A, img_T, mask_A, mask_T)
        # Outputs:
        #   - packages: (1,12,256,256)
        #   - color_inv_refs_pair: [Int, (1,3,64,64)], len=2
        #   - color_inv_refs_pair_cro: [Int, (1,3,64,64)], len=2
        pred_imgs = self.unet(packages)  # (1,3,256,256)
        return pred_imgs, packages, color_inv_refs_pair, color_inv_refs_pair_cro
