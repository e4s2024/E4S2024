import argparse
import cv2
import glob
import os

from PIL import Image
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

# from realesr.realesrgan import RealESRGANer
# from realesr.realesrgan.archs.srvgg_arch import SRVGGNetCompact


make_abs_path = lambda fn: os.path.join(os.path.dirname(os.path.realpath(__file__)), fn)


class EmptyArgs(object):
    def __init__(self):
        pass


class RealESRBatchInfer(object):
    def __init__(self):
        self.device = "cuda:0"
        self.args = EmptyArgs()
        self.args.model_name = "RealESRGAN_x4plus"
        self.args.model_path = make_abs_path("../../../ReliableSwap/pretrained/third_party/RealESRGAN/RealESRGAN_x4plus.pth")
        self.args.denoise_strength = 0.5
        self.args.face_enhance = False
        self.args.tile = 0
        self.args.gpu_id = "0"

        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        ).to(self.device)

        # prefer to use params_ema
        loadnet = torch.load(self.args.model_path, map_location=torch.device('cpu'))
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.model.load_state_dict(loadnet[keyname], strict=True)

        # self.upsampler = RealESRGANer(
        #     scale=4,
        #     model_path=self.args.model_path,
        #     dni_weight=None,
        #     model=self.model,
        #     gpu_id=self.args.gpu_id,
        # )

        print(f"[RealESRBatchInfer] loaded from {self.args.model_path}.")

    @torch.no_grad()
    def infer_batch(self, source_tensor: torch.Tensor, out_hw: tuple = None):
        if out_hw is None:
            out_hw = source_tensor.shape[2:]
        source_down = (source_tensor * 0.5 + 0.5).clamp(0, 1)
        source_down = F.interpolate(source_down, size=(256, 256), mode="bilinear", align_corners=True)
        result = self.model(source_down)  # (B,3,1024,1024)
        result = F.interpolate(result, size=out_hw, mode="bilinear", align_corners=True)
        result = (result * 2. - 1.).clamp(-1, 1)
        return result

    def infer_image(self, img: Image):
        img = np.array(img)
        img = torch.from_numpy(img).float().cuda()
        img = (img / 127.5) - 1.
        img = img.unsqueeze(0)
        img = rearrange(img, "n h w c -> n c h w").contiguous()
        res = self.infer_batch(img, out_hw=(1024, 1024))
        res = rearrange(res, "n c h w -> n h w c").contiguous()
        res = (res[0] * 127.5 + 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        return Image.fromarray(res)


if __name__ == "__main__":
    net = RealESRBatchInfer()
    img = torch.randn(2, 3, 1024, 1024).cuda()
    res = net.infer_batch(img)
    print(res.shape)
