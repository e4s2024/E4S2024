import torch
import torch.nn as nn
from swap_face_fine.LIA.networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args

        if args.model == 'vox':
            model_path = '/apdcephfs_cq2/share_1290939/branchwang/projects/LIA/checkpoints/vox.pt'
        elif args.model == 'taichi':
            model_path = 'checkpoints/taichi.pt'
        elif args.model == 'ted':
            model_path = 'checkpoints/ted.pt'
        else:
            raise NotImplementedError

        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        """
        print('==> loading data')
        self.save_path = args.save_folder + '/%s' % args.model
        os.makedirs(self.save_path, exist_ok=True)
        self.save_path = os.path.join(self.save_path, Path(args.source_path).stem + '_' + Path(args.driving_path).stem + '.mp4')
        self.img_source = img_preprocessing(args.source_path, args.size).cuda()
        self.vid_target, self.fps = vid_preprocessing(args.driving_path)
        self.vid_target = self.vid_target.cuda()
        """

    def run(self):

        print('==> running')
        with torch.no_grad():

            vid_target_recon = []

            if self.args.model == 'ted':
                h_start = None
            else:
                h_start = self.gen.enc.enc_motion(self.vid_target[:, 0, :, :, :])

            for i in tqdm(range(self.vid_target.size(1))):
                img_target = self.vid_target[:, i, :, :, :]
                img_recon = self.gen(self.img_source, img_target, h_start)
                vid_target_recon.append(img_recon.unsqueeze(2))

            vid_target_recon = torch.cat(vid_target_recon, dim=2)
            save_video(vid_target_recon, self.save_path, self.fps)


    def  run_online(self, source, target):
        # source_img = np.array(Image.open(source).resize((256, 256))) / 255.
        # target_img = np.array(Image.open(target).resize((256, 256))) / 255.
        source_img = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0) * 2. - 1.
        target_img = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0) * 2. - 1.
        source_img = source_img.cuda().float()
        target_img = target_img.cuda().float()

        with torch.no_grad():
            # h_start = None
            h_start = self.gen.enc.enc_motion(source_img)
            img_recon = self.gen(source_img, target_img, h_start)
        
        img_recon = img_recon[0].clamp(-1, 1).cpu().detach().permute(1, 2, 0)
        img_recon = (img_recon - img_recon.min()) / (img_recon.max() - img_recon.min())
        # img_recon = img_recon.numpy().astype(np.uint8)
        # Image.fromarray(img_recon).save(os.path.join(self.args.save_folder, 'recon.png'))

        return img_recon.numpy()



def drive_demo(source, target):
    args = lambda: None
    args.model = 'vox'
    args.size = 256
    args.channel_multiplier = 1
    args.latent_dim_style = 512
    args.latent_dim_motion = 20

    demo = Demo(args)

    return demo.run_online(source, target)



if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--source_path", type=str, default='')
    parser.add_argument("--driving_path", type=str, default='')
    parser.add_argument("--save_folder", type=str, default='./res')
    args = parser.parse_args()

    # demo
    demo = Demo(args)
    # demo.run()

    source = "/apdcephfs_cq2/share_1290939/branchwang/data/e4s_vis/f20_to_f19_faceVid2Vid/S_cropped.png"
    target = "/apdcephfs_cq2/share_1290939/branchwang/data/e4s_vis/f20_to_f19_faceVid2Vid/T_cropped.png"
    demo.run_online(source, target)
