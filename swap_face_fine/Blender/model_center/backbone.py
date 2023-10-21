import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .cmodules.base_network import BaseNetwork
from .cmodules.normalization import get_nonspade_norm_layer, equal_lr
from .cmodules.architecture import ResnetBlock as ResnetBlock
from .cmodules.architecture import SPADEResnetBlock as SPADEResnetBlock
from .cmodules.architecture import Attention


class AdaptiveFeatureGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        # TODO: kernel=4, concat noise, or change architecture to vgg feature pyramid
        super().__init__()
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw))
        if opt.warp_stride == 2:
            self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=1, padding=pw))
        else:
            self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

        nf = opt.ngf

        self.head_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        if opt.adaptor_nonlocal:
            self.attn = Attention(8 * nf, False)
        self.G_middle_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        self.G_middle_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt, use_se=opt.adaptor_se)

        if opt.adaptor_res_deeper:
            self.deeper0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
            if opt.dilation_conv:
                self.deeper1 = SPADEResnetBlock(4 * nf, 4 * nf, opt, dilation=2)
                self.deeper2 = SPADEResnetBlock(4 * nf, 4 * nf, opt, dilation=4)
                self.degridding0 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=2, dilation=2))
                self.degridding1 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=1))
            else:
                self.deeper1 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
                self.deeper2 = SPADEResnetBlock(4 * nf, 4 * nf, opt)

    def forward(self, input, seg):  # input:(1,3,256,256), seg:(1,3,256,256)=input
        x = self.layer1(input)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))  # (1,512,64,64)

        x = self.head_0(x, seg)
        if self.opt.adaptor_nonlocal:
            x = self.attn(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)  # (1,256,64,64)
        if self.opt.adaptor_res_deeper:
            x = self.deeper0(x, seg)
            x = self.deeper1(x, seg)
            x = self.deeper2(x, seg)
            if self.opt.dilation_conv:
                x = self.degridding0(x)
                x = self.degridding1(x)
        return x  # (1,256,64,64)


class SmallFPN(nn.Module):
    def __init__(self, ):
        super(SmallFPN, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 1, stride=2, padding=0)
        self.conv2 = nn.Conv2d(256, 256, 1, stride=2, padding=0)
        return

    def forward(self, x, y=None):
        return self.conv2(self.conv1(x))
