import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class InputEncodeLayer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(InputEncodeLayer, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        if ch_in != ch_out:
            self.sqz_layer = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.sqz_layer = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = x + self.sqz_layer(residual)
        return x


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(ch_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=True)

        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)

        if ch_in != ch_out:
            self.sqz_layer = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, padding=0, bias=True)
        else:
            self.sqz_layer = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = x + self.sqz_layer(residual)
        return x


class ResUNet(nn.Module):
    def __init__(self, args):
        super(ResUNet, self).__init__()

        if args.small_FPN:
            self.input_encoder_layer = InputEncodeLayer(12, 16)

            self.res_en_layer2 = ResBlock(16, 32, stride=2)
            self.res_en_layer3 = ResBlock(32, 64, stride=2)

            self.res_bridge_layer = ResBlock(64, 128, stride=2)

            self.res_de_layer3 = ResBlock(128 + 64, 64)
            self.res_de_layer2 = ResBlock(64 + 32, 32)
            self.res_de_layer1 = ResBlock(32 + 16, 16)

            self.output_decoder_layer = nn.Sequential(
                nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid()
            )
        else:
            self.input_encoder_layer = InputEncodeLayer(12, 64)

            self.res_en_layer2 = ResBlock(64, 128, stride=2)
            self.res_en_layer3 = ResBlock(128, 256, stride=2)

            self.res_bridge_layer = ResBlock(256, 512, stride=2)

            self.res_de_layer3 = ResBlock(512 + 256, 256)
            self.res_de_layer2 = ResBlock(256 + 128, 128)
            self.res_de_layer1 = ResBlock(128 + 64, 64)

            self.output_decoder_layer = nn.Sequential(
                nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid()
            )

    def forward(self, pkgs):
        en_feat1 = self.input_encoder_layer(pkgs)
        en_feat2 = self.res_en_layer2(en_feat1)
        en_feat3 = self.res_en_layer3(en_feat2)

        bridge_feat = self.res_bridge_layer(en_feat3)

        de_feat3 = self.res_de_layer3(torch.cat([F.upsample_bilinear(bridge_feat, scale_factor=2), en_feat3], dim=1))
        de_feat2 = self.res_de_layer2(torch.cat([F.upsample_bilinear(de_feat3, scale_factor=2), en_feat2], dim=1))
        de_feat1 = self.res_de_layer1(torch.cat([F.upsample_bilinear(de_feat2, scale_factor=2), en_feat1], dim=1))

        pred = self.output_decoder_layer(de_feat1)
        return pred
