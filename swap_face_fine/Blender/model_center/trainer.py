import os.path

import torch
from torchvision import models
import torch.nn as nn
from .blener import Blender
from pix2pix.models.networks import PixelDiscriminator, GANLoss
import torch.optim as optim
from loss_center.vgg_perception import PerceptualLoss
from utils.project_kits import denorm_img
import torch.nn.functional as F
from torch.optim import lr_scheduler

import traceback


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        self.netG = nn.DataParallel(Blender(args)).cuda()
        self.optimizer_G = optim.SGD(self.netG.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        self.scheduler_G = lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=args.iter_num, eta_min=0)

        self.optimizers = [self.optimizer_G]
        self.schedulers = [self.scheduler_G]
        self.model_names = ['netG']

        if args.lambda_VGG != 0:
            self.criterion_VGG = nn.DataParallel(PerceptualLoss()).cuda()

        if args.lambda_DIS != 0 or args.lambda_GAN != 0:
            self.netD = nn.DataParallel(PixelDiscriminator(input_nc=3, ndf=64)).cuda()
            self.optimizer_D = optim.SGD(self.netD.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
            self.criterion_GAN = GANLoss('vanilla').cuda()
            self.scheduler_D = lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=args.iter_num, eta_min=0)
            self.optimizers.append(self.optimizer_D)
            self.schedulers.append(self.scheduler_D)
            self.model_names.append('netD')

        return

    def set_input(self, img_A, img_T, mask_A, mask_T):
        self.img_A = img_A.cuda()
        self.img_T = img_T.cuda()
        self.mask_A = mask_A.cuda()
        self.mask_T = mask_T.cuda()

        self.img_GT = denorm_img(img_T).cuda()
        return

    def forward(self):
        self.img_pred, self.img_references, self.color_inv_refs_pair, self.color_inv_refs_pair_cro = \
            self.netG(self.img_A, self.img_T, self.mask_A, self.mask_T)

    def optimize_parameters(self):
        self.forward()

        if self.args.lambda_DIS != 0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            self.set_requires_grad(self.netD, False)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        return

    def backward_D(self):
        pred_fake = self.netD(self.img_pred.detach())
        self.loss_GAN_D_fake = self.criterion_GAN(pred_fake, False)
        pred_real = self.netD(self.img_GT)
        self.loss_GAN_D_real = self.criterion_GAN(pred_real, True)
        self.loss_GAN_D = (self.loss_GAN_D_fake + self.loss_GAN_D_real) * 0.5 * self.args.lambda_DIS
        self.loss_GAN_D.backward()

    def backward_G(self):
        self.total_loss = 0

        if self.args.lambda_L1 != 0:
            self.loss_L1 = F.l1_loss(self.img_pred, self.img_GT).mean()
            self.total_loss += self.args.lambda_L1 * self.loss_L1

        if self.args.lambda_VGG != 0:
            self.loss_VGG = self.criterion_VGG(self.img_pred, self.img_GT).mean()
            self.total_loss += self.args.lambda_VGG * self.loss_VGG

        if self.args.lambda_GAN != 0:
            pred_fake = self.netD(self.img_pred)
            self.loss_GAN_G = self.criterion_GAN(pred_fake, True)
            self.total_loss += self.args.lambda_GAN * self.loss_GAN_G

        if self.args.lambda_CYC != 0:
            A, B = self.color_inv_refs_pair
            self.loss_CYC = F.l1_loss(A, B).mean()
            self.total_loss += self.args.lambda_CYC * self.loss_CYC

        if self.args.lambda_CYC2 != 0:
            A, B = self.color_inv_refs_pair_cro
            self.loss_CYC2 = F.l1_loss(A, B).mean()
            self.total_loss += self.args.lambda_CYC2 * self.loss_CYC2

        self.total_loss.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_cur_loss_dict(self):
        loss_names = ['loss_L1', 'loss_VGG', 'loss_CYC', 'loss_CYC2', 'loss_GAN_D', 'loss_GAN_G']
        d = {}
        for name in loss_names:
            if hasattr(self, name):
                d[name] = eval(f'self.{name}.item()')
        return d

    def get_viz_pkg(self):
        viz_pkg = {}
        viz_pkg['img_A'] = self.img_A.cpu()
        viz_pkg['img_T'] = self.img_T.cpu()
        viz_pkg['mask_A'] = self.mask_A.cpu()
        viz_pkg['mask_T'] = self.mask_T.cpu()
        viz_pkg['img_references'] = self.img_references.cpu()
        viz_pkg['color_inv_refs'] = self.color_inv_refs_pair
        viz_pkg['img_pred'] = self.img_pred.cpu()
        viz_pkg['color_inv_refs_cro'] = self.color_inv_refs_pair_cro

        return viz_pkg

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        return old_lr, lr

    def save_model(self, batch_i, mlog=print):

        for name in self.model_names:
            if isinstance(name, str):
                save_path = f'output/{self.args.exp_name}/latest_{name}.pth'
                net = getattr(self, name)
                torch.save(net.module.cpu().state_dict(), save_path)
                mlog(f'saved: {save_path}')
                net = net.cuda()

        return

    def load_model(self, resume='', mlog=print):
        load_path = None

        if os.path.exists(resume):
            load_path = resume
        else:
            default_G_path = f'output/{self.args.exp_name}/latest_netG.pth'
            if os.path.exists(default_G_path):
                load_path = default_G_path

        if load_path is None:
            return
        else:
            netG_params = torch.load(load_path)
            self.netG.module.load_state_dict(netG_params)

            net_D_path = load_path.replace('netG', 'netD')
            if os.path.exists(net_D_path):
                netD_params = torch.load(net_D_path)
                self.netD.module.load_state_dict(netD_params)

            return
