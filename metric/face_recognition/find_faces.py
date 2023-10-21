import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from datasets.dataset import CelebAHQDataset, FolderDataset
from metric.face_recognition.arcface.iresnet import iresnet100

class FaceCompare(nn.Module):
    def __init__(self):
        super(FaceCompare, self).__init__()
        self.id_model = iresnet100()
        id_path = "/media/yuange/EXTERNAL_USB/weights/third_party/arcface/glint360k_cosface_r100_fp16_0.1/backbone.pth"
        weights = torch.load(id_path, map_location="cpu")
        self.id_model.load_state_dict(weights)
        self.id_model.eval()
        print('ID model loaded.')

        self.register_buffer(
            name="trans_matrix",
            tensor=torch.tensor(
                [
                    [
                        [1.07695457, -0.03625215, -1.56352194 / 512],
                        [0.03625215, 1.07695457, -5.32134629 / 512],
                    ]
                ],
                requires_grad=False,
            ).float(),
        )  # (1,2,3) # a horrible bug if not '/512', difference between Pytorch grid_sample and Kornia warp_affine

    @torch.no_grad()
    def forward(self, x):
        M = self.trans_matrix.repeat(x.size()[0], 1, 1)  # to (B,2,3)
        grid = F.affine_grid(M, size=x.size(), align_corners=True)  # 得到grid 用于grid sample
        x = F.grid_sample(x, grid, align_corners=True, mode="bilinear", padding_mode="zeros")  # warp affine

        x = F.interpolate(x, size=112, mode="bilinear", align_corners=True)
        return self.id_model(x)


if __name__ == "__main__":
    batch_size = 32
    device = torch.device("cuda:0")

    celeba_hq_folder = "/home/yuange/datasets/CelebA-HQ/"
    dataset_celebahq = CelebAHQDataset(
        celeba_hq_folder,
        mode="train"
    )
    dataloader1 = DataLoader(
        dataset_celebahq,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    cvpr2023_e4s_folder = "/home/yuange/datasets/CVPR2023_images/"
    dataset_cvpr2023 = FolderDataset(
        cvpr2023_e4s_folder
    )
    dataloader2 = DataLoader(
        dataset_cvpr2023,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    ffhq_folder = "/home/yuange/datasets/ffhq/images1024x1024/"
    dataset_ffhq = FolderDataset(
        ffhq_folder
    )
    dataloader3 = DataLoader(
        dataset_ffhq,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    face_comp = FaceCompare().to(device)

    feats1 = torch.randn((len(dataset_celebahq), 512), requires_grad=False).cuda()
    paths1 = []
    for idx, batch in enumerate(tqdm(dataloader1)):
        img = batch[0].to(device)
        img_path = batch[-1]
        feat = face_comp(img)

        feats1[idx * batch_size:
               min((idx + 1) * batch_size, feats1.shape[0])] = feat
        paths1.extend(img_path)
    print(len(paths1), paths1[-1])
    torch.save(feats1, "feats1.pth")

    feats2 = torch.randn((len(dataset_cvpr2023), 512), requires_grad=False).cuda()
    paths2 = []
    for idx, batch in enumerate(tqdm(dataloader2)):
        img = batch[0].to(device)
        img_path = batch[-1]
        feat = face_comp(img)

        feats2[idx * batch_size:
               min((idx + 1) * batch_size, feats2.shape[0])] = feat
        paths2.extend(img_path)
    print(len(paths2), paths2[-1])
    torch.save(feats2, "feats2.pth")

    feats3 = torch.randn((len(dataset_ffhq), 512), requires_grad=False).cuda()
    paths3 = []
    for idx, batch in enumerate(tqdm(dataloader3)):
        img = batch[0].to(device)
        img_path = batch[-1]
        feat = face_comp(img)

        feats3[idx * batch_size:
               min((idx + 1) * batch_size, feats3.shape[0])] = feat
        paths3.extend(img_path)
    print(len(paths3), paths3[-1])
    torch.save(feats3, "feats3.pth")

    for j in range(len(feats2)):
        vec2 = feats2[j]
        path2 = paths2[j]
        max_sim = -1.0
        max_path = ""
        for i in range(len(feats1)):
            vec1 = feats1[i]
            path1 = paths1[i]

            cos_sim = F.cosine_similarity(vec2.unsqueeze(0), vec1.unsqueeze(0))
            if cos_sim > max_sim:
                max_sim = cos_sim
                max_path = path1
        print(f"TARGET={path2}, FOUND: PATH={max_path}, MAX_COS_SIM={max_sim}.")

    for j in range(len(feats2)):
        vec2 = feats2[j]
        path2 = paths2[j]
        max_sim = -1.0
        max_path = ""
        for k in range(len(feats3)):
            vec3 = feats3[k]
            path3 = paths3[k]

            cos_sim = F.cosine_similarity(vec2.unsqueeze(0), vec3.unsqueeze(0))
            if cos_sim > max_sim:
                max_sim = cos_sim
                max_path = path3
        print(f"TARGET={path2}, FOUND: PATH={max_path}, MAX_COS_SIM={max_sim}.")
