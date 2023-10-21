
import torch
import torch.nn as nn
import torch.nn.functional as F

from swap_face_fine.Blender.model_center.blener import Blender
from torchvision import transforms
import numpy as np
from PIL import Image

from swap_face_fine.Blender.utils.parser import get_base_parser
from utils import torch_utils
from swap_face_fine.face_parsing.face_parsing_demo import faceParsing_demo
from gradio_utils.face_swapping import (
    get_facial_mask_from_seg19,
)


def add_hyper(parser):
    parser.add_argument('--lambda_L1', default=1., type=float)
    parser.add_argument('--lambda_VGG', default=1., type=float)

    parser.add_argument('--lambda_GAN', default=0., type=float)
    parser.add_argument('--lambda_DIS', default=0., type=float)

    parser.add_argument('--lambda_CYC', default=1., type=float)
    parser.add_argument('--lambda_CYC2', default=10., type=float)

    parser.add_argument("--small_FPN", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    return parser



def blending_two_faces(img_a, img_t, mask_a, mask_t):

    parser = get_base_parser()
    parser = add_hyper(parser)
    args = parser.parse_args()
    # args.eval_only = True
    # args.small_FPN = True

    netG = nn.DataParallel(Blender(args)).cuda()

    load_path = './pretrained/face_blender/latest_netG.pth'

    netG_params = torch.load(load_path)
    netG.module.load_state_dict(netG_params)

    netG.eval()

    img_a = img_a.resize((256, 256)).convert('RGB')
    img_t = img_t.resize((256, 256)).convert('RGB')

    mask_a = mask_a.resize((256, 256)).convert('L')
    mask_t = mask_t.resize((256, 256)).convert('L')

    to_tensor = transforms.ToTensor()
    to_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    mask_a = torch.tensor(np.array(mask_a)).unsqueeze(0).cuda()
    mask_t = torch.tensor(np.array(mask_t)).unsqueeze(0).cuda()

    img_a = to_norm(to_tensor(img_a)).unsqueeze(0).cuda()
    img_t = to_norm(to_tensor(img_t)).unsqueeze(0).cuda()

    with torch.no_grad():
        img_pred, _, _, _ = netG(img_a, img_t, mask_a, mask_t)

    img_pred = img_pred[0].permute(1, 2, 0).cpu().data.numpy()
    
    return Image.fromarray(np.uint8(img_pred * 255))


class BlenderInfer(object):
    def __init__(self):
        parser = get_base_parser()
        parser = add_hyper(parser)
        args = parser.parse_args()
        args.eval_only = True
        # args.small_FPN = True
        # print('[DEBUG inference] args:', args)

        netG = Blender(args).cuda()

        load_path = './pretrained/face_blender/latest_netG.pth'

        netG_params = torch.load(load_path)
        netG.load_state_dict(netG_params)
        netG.requires_grad_(False)
        netG.eval()

        self.netG = netG

    @torch.no_grad()
    def infer_image(self, img_a, img_t, mask_a, mask_t):
        """
        Transfer the color of img_t to img_a.
        """
        img_a = img_a.resize((256, 256)).convert('RGB')
        img_t = img_t.resize((256, 256)).convert('RGB')

        mask_a = mask_a.resize((256, 256)).convert('L')
        mask_t = mask_t.resize((256, 256)).convert('L')

        to_tensor = transforms.ToTensor()
        to_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        mask_a = torch.tensor(np.array(mask_a)).unsqueeze(0).cuda()
        mask_t = torch.tensor(np.array(mask_t)).unsqueeze(0).cuda()

        img_a = to_norm(to_tensor(img_a)).unsqueeze(0).cuda()
        img_t = to_norm(to_tensor(img_t)).unsqueeze(0).cuda()

        with torch.no_grad():
            img_pred, _, _, _ = self.netG(img_a, img_t, mask_a, mask_t)

        img_pred = img_pred[0].permute(1, 2, 0).cpu().data.numpy()

        return Image.fromarray(np.uint8(img_pred * 255))


class BlenderForTrain(nn.Module):
    def __init__(self,
                 enhance_model: nn.Module,
                 parsing_model: nn.Module,
                 mask_softer_model: nn.Module,
                 ):
        super().__init__()
        parser = get_base_parser()
        parser = add_hyper(parser)
        args = parser.parse_args()

        load_path = './pretrained/face_blender/latest_netG.pth'
        netG = Blender(args).cuda()
        netG_params = torch.load(load_path)
        netG.load_state_dict(netG_params)
        self.netG = netG

        self.enhance_model = enhance_model
        self.parsing_model = parsing_model
        self.mask_softer_model = mask_softer_model
        self.sobel = Sobel().cuda()

        self.freeze()

    def forward(self, tensor_a, tensor_t, mask_blending_from=None):
        """
        @param tensor_a: (B,RGB,1024,1024), in [-1,1], color to
        @param tensor_t: (B,RGB,1024,1024), in [-1,1], color from
        @param mask_blending_from: PIL.Image
        """
        with torch.no_grad():
            swapped_face_image = torch_utils.tensor2im(tensor_a)
            target_image = torch_utils.tensor2im(tensor_t)
            # target_tensor = torch_utils.im2tensor(T, std=True)
            swap_mask_19 = faceParsing_demo(self.parsing_model, swapped_face_image, convert_to_seg12=False)
            target_mask_19 = faceParsing_demo(self.parsing_model, target_image, convert_to_seg12=False)
            blending_mask = get_facial_mask_from_seg19(
                torch.LongTensor(swap_mask_19[None, None, :, :]),
                target_size=swapped_face_image.size, edge_softer=self.mask_softer_model, is_seg19=True
            )  # in [0,1]
            swap_mask_19 = torch_utils.im2tensor(swap_mask_19, add_c_dim=True, norm=False)
            target_mask_19 = torch_utils.im2tensor(target_mask_19, add_c_dim=True, norm=False)
            blending_mask = torch_utils.im2tensor(blending_mask, norm=False)

            mask_a = swap_mask_19
            mask_t = target_mask_19
            mask_blending = blending_mask

        out_hw = tensor_a.shape[2:]
        a = F.interpolate(tensor_a, size=(256, 256), mode="bilinear", align_corners=True)
        t = F.interpolate(tensor_t, size=(256, 256), mode="bilinear", align_corners=True)
        ma = F.interpolate(mask_a, size=(256, 256), mode="bilinear", align_corners=True)
        mt = F.interpolate(mask_t, size=(256, 256), mode="bilinear", align_corners=True)
        ma = ma[:, 0]  # squeeze C dim
        mt = mt[:, 0]  # squeeze C dim

        recolor, _, _, _ = self.netG(a, t, ma, mt)  # (B,3,256,256), in [0,1]
        recolor = recolor.clamp(0, 1)

        ''' super-res '''
        recolor = self.enhance_model(recolor)  # input:[0,1], output:[0,1]
        recolor = F.interpolate(recolor, size=out_hw, mode="bilinear", align_corners=True)
        recolor = (recolor * 2. - 1.).clamp(-1, 1)

        ''' alpha blending '''
        edge = self.sobel(recolor)
        no_edge_mask = (mask_blending - edge).clip(0, 1)
        alpha_no_edge_mask = 0.95 * no_edge_mask
        recolor = tensor_a * (1 - alpha_no_edge_mask) + recolor * alpha_no_edge_mask

        return (recolor.clip(-1, 1),  # (B,RGB,1024,1024), in [-1,1]
                no_edge_mask)

    def freeze(self):
        self.netG.requires_grad_(False)
        self.enhance_model.requires_grad_(False)
        self.parsing_model.requires_grad_(False)
        self.mask_softer_model.requires_grad_(False)
        self.sobel.requires_grad_(False)
        unfreeze_cnt = 0
        for param in self.parameters():
            if param.requires_grad:
                unfreeze_cnt += 1
        print(f"[BlenderForTrain] frozen: trainable={unfreeze_cnt}")

    def unfreeze(self):
        self.netG.unet.requires_grad_(True)  # only tuning unet
        unfreeze_cnt = 0
        for param in self.parameters():
            if param.requires_grad:
                unfreeze_cnt += 1
        print(f"[BlenderForTrain] unfrozen: trainable={unfreeze_cnt}")

    def save_weights(self, to_path: str):
        torch.save(self.netG.state_dict(), to_path)


class Sobel(nn.Module):
    def __init__(self, in_channels = 3,
            out_channels = 1):
        super().__init__()
        conv_op = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        sobel_kernel = np.repeat(sobel_kernel, in_channels, axis=1)
        sobel_kernel = np.repeat(sobel_kernel, out_channels, axis=0)

        conv_op.weight.data = torch.from_numpy(sobel_kernel)

        self.conv_op = conv_op

    def forward(self, im):
        edge_detect = self.conv_op(im)
        return edge_detect  # in [0,1]


if __name__ == "__main__":

    i_a = torch.randn(1, 3, 256, 256).cuda()
    i_t = torch.rand_like(i_a).cuda()
    m_a = torch.randn(1, 256, 256).cuda() * 10
    m_b = torch.rand_like(m_a).cuda() * 10

    parser = get_base_parser()
    parser = add_hyper(parser)
    args = parser.parse_args()
    net = Blender(args).cuda()

    y = net(i_a, i_t, m_a, m_b)
