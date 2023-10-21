import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module, InstanceNorm2d

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE, bottleneck_IR_SE_Ours
from models.stylegan2.model import EqualLinear, EqualConv2d
from models.encoders.helpers import get_block

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x
    
class CustomBackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(CustomBackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        # print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = 11
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        
        self.structure_linear = EqualConv2d(256, 512, 1)
                                        
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        
        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 20:
                structure_feats = x
            
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        
        structure_feats = self.structure_linear(structure_feats)
        return x, structure_feats
    
    
# ================================================================
class FSEncoder_SEAN(Module):
    """ code adapted from the Zencoder from SEAN paper.

        https://github.com/ZPdesu/SEAN/blob/master/models/networks/architecture.py
    
    
    基本上，就是用一个网络从 1024 -> 32 的分辨率，
        (1) 其中， 分辨率为128的中间表达作为 styleGAN的底部11层的 style code;
        (2) 而32分辨率的特征则对应学一个到 styleGAN 前7层的平均 feature map的残差
    
    """
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=3, norm_layer=nn.InstanceNorm2d, in_size = 1024):
        super(FSEncoder_SEAN, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0),
                 norm_layer(ngf), nn.LeakyReLU(0.2, False)]
        
        if in_size ==256:
            n_downsampling = 2
             
        # downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, False)]

        # SEAN 的RGB输入为256*256，所以最后用了一个上采样，确保feature maps的输出是128*128的。
        # 我们这里最后一步暂时不用上采样，因为我们的图片输入是 1024*1024的
        if in_size == 256:
            ### upsample
            for i in range(1):
                mult = 2**(n_downsampling - i)
                model += [nn.ConvTranspose2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(int(ngf * mult / 2)), nn.LeakyReLU(0.2, False)]  # TODO: 很奇怪，为啥SEAN的代码里面，instanceNorm2d定义时的channel数和输入的特征的通道数不匹配
        
        self.model = nn.Sequential(*model)  # 学到 128 尺度的中间 特征表达
        
        # 学 region 对应的 style code
        style_module = [nn.ReflectionPad2d(1), nn.Conv2d(256, output_nc, kernel_size=3, padding=0), nn.Tanh()]
        self.style_module = nn.Sequential(*style_module)
        
        # 额外的模块 学一个到 styleGAN 前7/5层的平均 feature map的残差
        structure_module = [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), norm_layer(512), nn.LeakyReLU(0.2, False)]            
        structure_module += [nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),norm_layer(512), nn.LeakyReLU(0.2, False)]
        structure_module += [nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),norm_layer(512), nn.LeakyReLU(0.2, False)]
        # structure_module += [nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),norm_layer(512), nn.LeakyReLU(0.2, False)]
        self.structure_module = nn.Sequential(*structure_module)
        
    def forward(self, input, segmap):
        # 共享的特征
        feats = self.model(input)  # [bs,C,H,W]
        
        # 1. 学 style code
        codes = self.style_module(feats)
        
        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')
        b_size = codes.shape[0]
        f_size = codes.shape[1]
        s_size = segmap.shape[1]  # seg 类别数

        codes_vector = torch.zeros(
            (b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)  # 每个mask label 都对应有一个 style code

        for i in range(b_size):  # 每个sample
            for j in range(s_size):  # 每个seg cls
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:  # 如果某个cls存在对应的像素，做mask avg pooling，否则为0
                    codes_component_feature = codes[i].masked_select(
                        segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

                    # codes_avg[i].masked_scatter_(segmap.bool()[i, j], codes_component_mu)

        # 2. 学 structure code
        structure_feats = self.structure_module(feats)
        
        return codes_vector, structure_feats


class FSEncoder_PSP(Module):
    def __init__(self, mode='ir_se', opts=None):
        super(FSEncoder_PSP, self).__init__()
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = [
            get_block(in_channel=64, depth=128, num_units=3),
			get_block(in_channel=128, depth=256, num_units=4),
			get_block(in_channel=256, depth=512, num_units=14),
			get_block(in_channel=512, depth=512, num_units=3)
		]
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE_Ours
        self.n_styles = 11
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      InstanceNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        # # 16 * 16 尺度的 structure code
        # self.structure_linear = Sequential(
        #     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.PReLU(num_parameters=512),
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )
    
    def get_per_comp_styleCode(self, style_feats, segmap):
        """根据输入的 feature map 得到各个component的 style code"""    
        segmap = F.interpolate(segmap, size=style_feats.size()[2:], mode='nearest')

        b_size = style_feats.shape[0]
        f_size = style_feats.shape[1]
        s_size = segmap.shape[1]  # seg 类别数

        codes_vector = torch.zeros(
            (b_size, s_size, f_size), dtype=style_feats.dtype, device=style_feats.device)  # 每个mask label 都对应有一个 style code

        for i in range(b_size):  # 每个sample
            for j in range(s_size):  # 每个seg cls
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:  # 如果某个cls存在对应的像素，做mask avg pooling，否则为0
                    codes_component_feature = style_feats[i].masked_select(
                        segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

        return codes_vector
        
    def forward(self, x, segmap):
        x = self.input_layer(x)
        
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                s1 = x
            elif i==20:
                s2 = x
            elif i == 23:
                s3 = x
                
        # # (1) 处理 structure code  
        # structure_feats = self.structure_linear(x)
        structure_feats = torch.zeros_like(x)
        
        # (2) 处理 style code  
        code_vectors1 = self.get_per_comp_styleCode(s1,segmap)
        code_vectors2 = self.get_per_comp_styleCode(s2,segmap)
        code_vectors3 = self.get_per_comp_styleCode(s3,segmap)
        
        codes_vector = torch.cat([code_vectors1, code_vectors2, code_vectors3], dim=2)

        return codes_vector, structure_feats