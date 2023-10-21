import os

import cv2.gapi
from PIL import Image
import PIL.ImageQt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


""" From MegaFS: https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels/tree/main/inference """
class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


class Trick(object):
    def __init__(self):
        self.gpen_model = None
        self.mouth_helper = None

    @staticmethod
    def get_any_mask(img, par=None, normalized=False):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
        # 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ori_h, ori_w = img.shape[2], img.shape[3]
        with torch.no_grad():
            img = F.interpolate(img, size=512, mode="nearest", )
            if not normalized:
                img = img * 0.5 + 0.5
                img = img.sub(vgg_mean.detach()).div(vgg_std.detach())
            out = global_bisenet(img)[0]
            parsing = out.softmax(1).argmax(1)
        mask = torch.zeros_like(parsing)
        for p in par:
            mask = mask + ((parsing == p).float())
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(ori_h, ori_w), mode="bilinear", align_corners=True)
        return mask

    @staticmethod
    def finetune_mask(facial_mask: np.ndarray, lmk_98: np.ndarray = None):
        scale_ratio = facial_mask.shape[1] / 256
        facial_mask = (facial_mask * 255).astype(np.uint8)
        # h_min = lmk_98[33:41, 0].min() + 20
        h_min = 80 * scale_ratio

        facial_mask = cv2.dilate(facial_mask, (40, 40), iterations=1)
        facial_mask[:h_min] = 0  # black
        facial_mask[(255 - 20) * scale_ratio:] = 0

        kernel_size = (20, 20)
        blur_size = tuple(2 * j + 1 for j in kernel_size)
        facial_mask = cv2.GaussianBlur(facial_mask, blur_size, 0)

        return facial_mask.astype(np.float) / 255

    @staticmethod
    def smooth_mask(mask_tensor: torch.Tensor):
        mask_tensor, _ = global_smooth_mask(mask_tensor)
        return mask_tensor

    @staticmethod
    def tensor_to_arr(tensor):
        return ((tensor + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    @staticmethod
    def arr_to_tensor(arr, norm: bool = True):
        tensor = torch.tensor(arr, dtype=torch.float).cuda() / 255  # in [0,1]
        tensor = (tensor - 0.5) / 0.5 if norm else tensor  # in [-1,1]
        tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def gpen(self, img_np: np.ndarray, use_gpen=True):
        if not use_gpen:
            return img_np
        if self.gpen_model is None:
            self.gpen_model = GPENImageInfer()
        img_np = self.gpen_model.image_infer(img_np)
        return img_np

    def finetune_mouth(self, i_s, i_t, i_r):
        if self.mouth_helper is None:
            self.load_mouth_helper()
        helper_face = self.mouth_helper(i_s, i_t)[0]
        i_r_mouth_mask = self.get_any_mask(i_r, par=[11, 12, 13])  # (B,1,H,W)

        ''' dilate and blur by cv2 '''
        i_r_mouth_mask = self.tensor_to_arr(i_r_mouth_mask)[0]  # (H,W,C)
        i_r_mouth_mask = cv2.dilate(i_r_mouth_mask, (20, 20), iterations=1)

        kernel_size = (5, 5)
        blur_size = tuple(2 * j + 1 for j in kernel_size)
        i_r_mouth_mask = cv2.GaussianBlur(i_r_mouth_mask, blur_size, 0)  # (H,W,C)
        i_r_mouth_mask = i_r_mouth_mask.squeeze()[None, :, :, None]  # (1,H,W,1)
        i_r_mouth_mask = self.arr_to_tensor(i_r_mouth_mask, norm=False)  # in [0,1]

        return helper_face * i_r_mouth_mask + i_r * (1 - i_r_mouth_mask)

    @staticmethod
    def blending_two_images_with_mask(bottom: Image, up: Image,
                                      up_ratio: float = 1.0,
                                      up_mask: np.ndarray = None
                                      ) -> Image:
        h, w = bottom.size
        if up_mask is None:
            up_mask = np.ones((h, w, 1), dtype=float)
        else:
            up_mask = up_mask.squeeze()[:, :, None]
        up_mask[np.isnan(up_mask)] = 0.  # input may contain NAN
        assert 0.0 <= up_ratio <= 1.0, "Blending Ratio should be in [0.0, 1.0]!"
        up_mask *= up_ratio
        i_a = np.array(bottom)
        i_b = np.array(up)
        i_a = i_a * (1 - up_mask) + i_b * up_mask
        ret_image = Image.fromarray(i_a.astype(np.uint8).clip(0, 255))
        return ret_image

    @staticmethod
    def sharp_image(img: Image, sharp_k: int = 10) -> Image:
        img = np.array(img)
        gau = cv2.GaussianBlur(img, (0, 0), sharp_k)
        dst = cv2.addWeighted(img, 1.5, gau, -0.5, 0)
        return Image.fromarray(dst.astype(np.uint8).clip(0, 255))

    @staticmethod
    def get_edge(img: Image, threshold: int = 128) -> Image:
        img = np.array(img).astype(np.uint8)
        img_x = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
        img_y = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))
        edge = cv2.addWeighted(img_x, 1, img_y, 1, 0)
        edge = cv2.cvtColor(edge, cv2.COLOR_RGB2GRAY)
        pos_big = np.where(edge >= threshold)
        pos_small = np.where(edge < threshold)
        # edge[pos_big] = (edge[pos_big] * 1.75).clip(0, 255)
        edge = cv2.GaussianBlur(edge, ksize=(3, 3), sigmaX=5)
        edge[pos_big] = (edge[pos_big] * 1.05).clip(0, 255)
        edge[pos_small] = (edge[pos_small] / 1.).clip(0, 255)
        edge = cv2.GaussianBlur(edge, ksize=(5, 5), sigmaX=11)
        return Image.fromarray(edge.astype(np.uint8))

    @staticmethod
    def get_facial_mask_from_seg19(seg_map_long: torch.LongTensor,
                                   target_size: tuple = None,
                                   edge_softer: SoftErosion = None,
                                   ):
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
        facial_indices = (1, 2, 3, 5, 6, 8, 9)
        mask = torch.zeros_like(seg_map_long, dtype=torch.long)
        for index in facial_indices:
            mask = mask + ((seg_map_long == index).long())  # either in {0,1}
        mask = mask.float()
        if target_size is not None:
            mask = F.interpolate(mask, size=target_size, mode="bilinear", align_corners=True)
        if edge_softer is not None:
            mask, _ = edge_softer(mask)
        return mask.cpu().numpy()


# vgg_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]],
#                              requires_grad=False, device=torch.device(0))
# vgg_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]],
#                             requires_grad=False, device=torch.device(0))
# def load_bisenet():
#     bisenet_model = BiSeNet(n_classes=19)
#     bisenet_model.load_state_dict(
#         torch.load(make_abs_path("/gavin/datasets/hanbang/79999_iter.pth",), map_location="cpu")
#     )
#     bisenet_model.eval()
#     bisenet_model = bisenet_model.cuda(0)
#
#     smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
#     print('[Global] bisenet loaded.')
#     return bisenet_model, smooth_mask
#
# global_bisenet, global_smooth_mask = load_bisenet()

if __name__ == "__main__":
    img_down = "Step1_gen_swapped_face.png"
    img_up = "1024x_Step2_color_transfer_enhance.png"
    img_down = Image.open(img_down).convert("RGB")
    img_up = Image.open(img_up).convert("RGB")

    img_edge = Trick.get_edge(img_down)
    # img_blend = Trick.blending_two_images_with_mask(
    #
    # )
    img_edge.save("tmp_edge.png")

    img_edge = np.array(img_edge).astype(np.float32) / 255.
    img_masked_down = img_edge[:, :, None] * np.array(img_down).astype(np.float32)
    img_masked_down = img_masked_down.clip(0, 255).astype(np.uint8)
    Image.fromarray(img_masked_down).save("tmp_masked_down.png")

    img_masked_up = (1 - img_edge[:, :, None]) * np.array(img_up).astype(np.float32)
    img_masked_up = img_masked_up.clip(0, 255).astype(np.uint8)
    Image.fromarray(img_masked_up).save("tmp_masked_up.png")

    img_blend = (img_masked_down + img_masked_up).clip(0, 255)
    Image.fromarray(img_blend).save("tmp_blend.png")
