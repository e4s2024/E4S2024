import os.path as osp
import numpy as np
from PIL import Image
import cv2

import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset


def recolor(x: Image):
    trans1 = transforms.Grayscale()
    trans2 = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    x = trans2(x)
    return x


def flip(x: Image):
    trans1 = transforms.RandomHorizontalFlip(p=1.0)
    x = trans1(x)
    return x


if __name__ == "__main__":
    img_path = "../outputs/28049_to_28462/T_cropped.png"
    img = Image.open(img_path).convert("RGB")

    img1 = recolor(img)
    img2 = flip(img)

    img1.save("tmp1_recolor.png")
    img2.save("tmp2_flip.png")
