import argparse

import torch
import numpy as np
from PIL import Image

from swap_face_fine.SwinIR.main_test_swinir import test, define_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                                                             'Just used to differentiate two different settings in Table 2 of the paper. '
                                                                             'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    """ additional settings """
    args.task = "real_sr"
    args.model_path = "./pretrained/SwinIR/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    args.scale = 4

    return args


class SwinIRInfer(object):
    def __init__(self, device: torch.device):
        self.device = device
        self.window_size = 8
        self.args = get_args()

        self.model = define_model(self.args)
        self.model.eval()
        self.model = self.model.to(device)

    @torch.no_grad()
    def infer(self, img_lq: Image):
        # imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.array(img_lq).astype(np.float32) / 255.
        img_lq = np.transpose(img_lq, (2, 0, 1))  # HCW-RGB to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            window_size = self.window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, self.model, self.args, window_size)
            output = output[..., :h_old * self.args.scale, :w_old * self.args.scale]

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[:, :, :], (1, 2, 0))  # CHW-RGB to HCW-RGB
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        output = Image.fromarray(output)

        return output


if __name__ == "__main__":
    lo_img_path = "./outputs/4829_to_5982/256x_Step3b_inpainting_generated.png"
    hi_img_path = "./tmp_swinir.png"

    swin_ir_infer = SwinIRInfer(
        device=torch.device("cuda:0")
    )
    hi_img = swin_ir_infer.infer(Image.open(lo_img_path, ).convert("RGB"))
    hi_img.save(hi_img_path)
