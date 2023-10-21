import json
import os
import random

import numpy as np
import scipy
import torch
import torchvision.transforms.functional as F
from PIL import Image
# from scipy.misc import imread
from imageio import imread
from skimage.color import rgb2gray, gray2rgb

from torch.utils.data import DataLoader



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = []
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMSMASK_REVERSE

        self.reverse_mask = config.MASK_REVERSE
        self.mask_threshold = config.MASK_THRESHOLD

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

        print('training:{}  mask:{}  mask_list:{}  data_list:{}'.format(training, self.mask, mask_flist, flist))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image
        img_gray = rgb2gray(img)

        # load mask
        mask = self.load_mask(img, index % len(self.mask_data))

        if self.reverse_mask == 1:
            mask = 255 - mask


        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            mask = mask[:, ::-1, ...]


        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(img_gray), self.to_tensor(mask)


    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        if self.training:
            mask_index = random.randint(0, len(self.mask_data) - 1)
        else:
            mask_index = index
            print('+++++++++++++++')

        mask = imread(self.mask_data[mask_index])
        mask = self.resize(mask, imgh, imgw)
        mask = (mask > self.mask_threshold).astype(np.uint8) * 255       # threshold due to interpolation


        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if flist is None:
            return []
        with open(flist, 'r') as j:
            f_list = json.load(j)
            return f_list


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
