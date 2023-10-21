"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm

# 我添加的
from models.stylegan2.model import EqualLinear
from models.stylegan2.model import Generator

import math
from models.encoders import psp_encoders
from models.encoders.psp_encoders import FSEncoder_PSP,FSEncoder_SEAN
from models.encoders.helpers import get_block, Flatten, bottleneck_IR, bottleneck_IR_SE
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
from models.stylegan2.model import EqualLinear, EqualConv2d

class LocalMLP(nn.Module):
    """将local components 的style code 转到styleGAN 的 W+ space， 即 512--> 18*512"""
    def __init__(self, dim_component=512, dim_style=512, num_w_layers=18,latent_squeeze_ratio=1):
        super(LocalMLP, self).__init__()
        self.dim_component = dim_component
        self.dim_style = dim_style
        self.num_w_layers = num_w_layers

        # 再细看一下Equalized Linear层
        self.mlp = nn.Sequential(
            EqualLinear(dim_component, dim_style//latent_squeeze_ratio, lr_mul=1),
            nn.LeakyReLU(),
            EqualLinear(dim_style//latent_squeeze_ratio, dim_style*num_w_layers, lr_mul=1)
        )

    def forward(self, x):
        """将local components 的style code 转到styleGAN 的 W+ space

        Args:
            x (torch.Tensor): with shape [bs, dim_component]

        Returns:
            out: [bs,18,512]
        """
        out=self.mlp(x)
        out=out.view(-1,self.num_w_layers,self.dim_style) # [bs,18,512]
        return out

class Net3(nn.Module):
    """ FSEncoder + styleGAN2 """

    def __init__(self,opts,):
        super(Net3, self).__init__()
        self.opts=opts
        assert self.opts.fsencoder_type in ["psp","sean"]
        if self.opts.fsencoder_type=="psp":
            self.encoder = FSEncoder_PSP(mode='ir_se', opts=self.opts)
            dim_s_code = 256 + 512 + 512
        else:
            self.encoder = FSEncoder_SEAN(input_nc=3, output_nc=512,in_size = 256)
            dim_s_code = 512
        
        self.split_layer_idx = 5
        self.remaining_layer_idx = self.opts.remaining_layer_idx
        
        # 区分component 的 W+ space 的 MLPs
        self.MLPs = nn.ModuleList()
        for i in range(self.opts.num_seg_cls):
            self.MLPs.append(
                LocalMLP(
                    dim_component=dim_s_code,
                    dim_style=512,
                    num_w_layers= self.remaining_layer_idx if self.remaining_layer_idx != 17 else 18
                )
            )
   
        self.G = Generator(size=self.opts.out_size, style_dim=512, n_mlp=8, split_layer_idx = self.split_layer_idx, remaining_layer_idx = self.remaining_layer_idx)

        # styleGAN的参数是否更新
        if not self.opts.train_G:
            for param in self.G.parameters():
                param.requires_grad = False
        # 注意，styleGAN的8层FC是永远不更新的
        else:
            for param in self.G.style.parameters():
                param.requires_grad = False
                
        # styleGAN的倒数几层不更新 (包括convs 和 ToRGBs)
        if self.remaining_layer_idx != 17:
            for param in self.G.convs[-(17-self.remaining_layer_idx):].parameters():
                param.requires_grad = False
            for param in self.G.to_rgbs[-(17-self.remaining_layer_idx)//2 - 1:].parameters():
                param.requires_grad = False
            
    
    def forward(self, img,mask, resize=False, randomize_noise=True,return_latents=False):
        """输入一张RGB图和对应的mask,
        (1) encoder 得到对应的F/S空间的特征，
        (2) 再送到styleGAN得到一张输出的图片

        Args:
            img (Tensor): 一对RGB图, each with shape [bs,3,1024,1024]
            mask ([type]):  一对RGB图对应的mask图, each with shape [bs,#seg_cls,1024,1024]
            resize (bool, optional): G生成的图片是否 resize. Defaults to True.
            randomize_noise (bool, optional): 是否加入随机噪声. Defaults to True.
            return_latents (bool, optional): 是否返回style codes. Defaults to False.

        Returns:
            [type]: [description]
        """
        if self.opts.fsencoder_type=="psp":
            codes_vector, structure_feats = self.encoder(F.interpolate(img,(256,256),mode='bilinear'), mask)  # [bs,#seg_cls, D], [bs,C,32,32]
        else:
            codes_vector, structure_feats = self.encoder(F.interpolate(img,(256,256),mode='bilinear'), mask)  # [bs,#seg_cls, D], [bs,C,32,32]
        codes=[]
        bs, num_comp = codes_vector.size(0), codes_vector.size(1)
        for i in range(num_comp):
            codes.append(self.MLPs[i](codes_vector[:,i,:])) 
        codes=torch.stack(codes,dim=1)   # [bs, #seg_cls, 13, 512]
        
        
        # # 剩下的几层不用分component
        # remaining_codes=[]
        # for i in range(len(self.remain_MLPs)):
        #     remaining_codes.append(self.remain_MLPs[i](codes_vector.view(bs, -1)))
        # remaining_codes = torch.stack(remaining_codes,dim=1)  #  [bs,5,512]

        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                 # 为了保持接口统一，将后3层的 style code 也扩展出一个 #seg_cls 维度
                codes = codes + self.latent_avg[:self.remaining_layer_idx, :].repeat(codes.shape[0],codes.shape[1],1)
                remaining_codes =  self.latent_avg[self.remaining_layer_idx:, :].repeat(bs, num_comp, 1) 
                codes = torch.cat([codes, remaining_codes],dim=2)
            else:
                if self.remaining_layer_idx != 17:
                    codes = codes + self.latent_avg[:self.remaining_layer_idx, :].repeat(codes.shape[0],codes.shape[1],1, 1)
                    remaining_codes =  self.latent_avg[self.remaining_layer_idx:, :].repeat(bs, num_comp, 1, 1)    
                    codes = torch.cat([codes, remaining_codes],dim=2)
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0],codes.shape[1],1, 1)
                
        # 1. 完全使用 style code  i.e., G(w)
        images1, result_latent, structure_feats_GT = self.G([codes], structure_feats, mask, input_is_latent=True,
                                                            randomize_noise=randomize_noise,return_latents=return_latents,
                                                            use_structure_code=False)
    
        
        # # 2. 使用 style code 和 strcture code i.e., G(w,F)
        # images2, _ , _ = self.G([codes], structure_feats, mask, input_is_latent=True,
        #                                                     randomize_noise=randomize_noise,return_latents=return_latents,
        #                                                     use_structure_code=True)
        
        if return_latents:
            return images1, structure_feats_GT, result_latent
        else:
            return images1, structure_feats_GT

    def get_style(self, img, mask):
        """输入一张RGB图和对应的mask, 得到各个component 对应的style codes
        
        Args:
            img (Tensor): RGB图, each with shape [bs,3,1024,1024]
            mask (Tensor):  RGB图对应的mask图, each with shape [bs,#seg_cls,1024,1024]
           
        Returns:
            structure_feats(Tensor): 图片的structure code, with shape [bs,512,32,32], 注意，这里其实是相对于StyleGAN第层输出的残差
            all_codes(Tensor): 各个component 对应的style codes, with shape [bs,#comp,18,512]。
                               !!! 注意，前7层的各个compnent其实没有意义，只是为了统一接口让shape保持一致,用的时候只用第1个即可 !!!
        """
        if self.opts.fsencoder_type=="psp":
            codes_vector, structure_feats = self.encoder(F.interpolate(img,(256,256),mode='bilinear'), mask)  # [bs,#seg_cls, D], [bs,C,32,32]
        else:
            codes_vector, structure_feats = self.encoder(F.interpolate(img,(256,256),mode='bilinear'), mask)  # [bs,#seg_cls, D], [bs,C,32,32]
        codes=[]
        bs, num_comp = codes_vector.size(0), codes_vector.size(1)
        for i in range(num_comp):
            codes.append(self.MLPs[i](codes_vector[:,i,:])) 
        codes=torch.stack(codes,dim=1)   # [bs, #seg_cls, 11,512]

        # # 剩下的几层不用分component
        # remaining_codes=[]
        # for i in range(len(self.remain_MLPs)):
        #     remaining_codes.append(self.remain_MLPs[i](codes_vector.view(bs, -1)))
        # remaining_codes = torch.stack(remaining_codes,dim=1)  #  [bs,5,512]

        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                 # 为了保持接口统一，将后3层的 style code 也扩展出一个 #seg_cls 维度
                codes = codes + self.latent_avg[:self.remaining_layer_idx, :].repeat(codes.shape[0],codes.shape[1],1)
                remaining_codes =  self.latent_avg[self.remaining_layer_idx:, :].repeat(bs, num_comp, 1) 
                style_codes = torch.cat([codes, remaining_codes],dim=2)
            else:
                if self.remaining_layer_idx != 17:
                    codes = codes + self.latent_avg[:self.remaining_layer_idx, :].repeat(codes.shape[0],codes.shape[1],1, 1)
                    remaining_codes =  self.latent_avg[self.remaining_layer_idx:, :].repeat(bs, num_comp, 1, 1)    
                    style_codes = torch.cat([codes, remaining_codes],dim=2)
                else:
                    style_codes = codes + self.latent_avg.repeat(codes.shape[0],codes.shape[1],1, 1)
            
        return structure_feats, style_codes

    def get_style_vectors(self, img, mask):
        """输入一张RGB图和对应的mask, 得到各个component 对应的style vectors
        
        Args:
            img (Tensor): RGB图, each with shape [bs,3,1024,1024]
            mask (Tensor):  RGB图对应的mask图, each with shape [bs,#seg_cls,1024,1024]
           
        Returns:
            style_vectors(Tensor): with shape [bs,#seg_cls,512]
        """
        if self.opts.fsencoder_type=="psp":
            style_vectors, structure_feats = self.encoder(F.interpolate(img,(256,256),mode='bilinear'), mask)  # [bs,#seg_cls, D], [bs,C,32,32]
        else:
            style_vectors, structure_feats = self.encoder(F.interpolate(img,(256,256),mode='bilinear'), mask)  # [bs,#seg_cls, D], [bs,C,32,32]
        
        return style_vectors, structure_feats
    
    def cal_style_codes(self,style_vectors):
        """根据每个compnent的 style vector转到styleGAN的style code"""
        
        codes=[]
        bs, num_comp = style_vectors.size(0), style_vectors.size(1)
        for i in range(num_comp):
            codes.append(self.MLPs[i](style_vectors[:,i,:])) 
        codes=torch.stack(codes,dim=1)   # [bs, #seg_cls, 11,512]

        # # 剩下的几层不用分component
        # remaining_codes=[]
        # for i in range(len(self.remain_MLPs)):
        #     remaining_codes.append(self.remain_MLPs[i](style_vectors.view(bs, -1)))
        # remaining_codes = torch.stack(remaining_codes,dim=1)  #  [bs,5,512]

        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                 # 为了保持接口统一，将后3层的 style code 也扩展出一个 #seg_cls 维度
                codes = codes + self.latent_avg[:self.remaining_layer_idx, :].repeat(codes.shape[0],codes.shape[1],1)
                remaining_codes =  self.latent_avg[self.remaining_layer_idx:, :].repeat(bs, num_comp, 1) 
                style_codes = torch.cat([codes, remaining_codes],dim=2)
            else:
                if self.remaining_layer_idx != 17:
                    codes = codes + self.latent_avg[:self.remaining_layer_idx, :].repeat(codes.shape[0],codes.shape[1],1, 1)
                    remaining_codes =  self.latent_avg[self.remaining_layer_idx:, :].repeat(bs, num_comp, 1, 1)    
                    style_codes = torch.cat([codes, remaining_codes],dim=2)
                else:
                    style_codes = codes + self.latent_avg.repeat(codes.shape[0],codes.shape[1],1, 1)
          
        return style_codes

    def gen_img(self, struc_codes, style_codes, mask, randomize_noise=True, noise=None, return_latents=False):
        """输入一张mask 和 对应各components的style codes，以及这张图片的structure code, 生成一张图片
        
        Args:
            style_codes (Tensor): 各个component 对应的style codes, with shape [bs,#comp,18,512]
            struc_codes (Tensor)
            mask (Tensor):  mask图, with shape [bs,#seg_cls,1024,1024]
            
            randomize_noise (bool, optional): 是否加入随机噪声. Defaults to True.
            return_latents (bool, optional): 是否返回style codes. Defaults to False.

        Returns:
            [type]: [description]
        """
        
        images, result_latent, structure_feats = self.G([style_codes], struc_codes, mask, input_is_latent=True,
                                       randomize_noise=randomize_noise,noise=noise,return_latents=return_latents,
                                       use_structure_code=False)

        if return_latents:
            return images, result_latent, structure_feats
        else:
            return images,-1, structure_feats
        
        