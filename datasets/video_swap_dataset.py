from datasets.dataset import TO_TENSOR, NORMALIZE
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

# 视频换脸 finetune G 时需要的 dataset
class VideoFaceSwappingDataset(Dataset):
    def __init__(self, driven, driven_recolor, driven_mask, driven_style_vector,
                 target, target_mask, target_style_vector,
                 img_transform=TO_TENSOR, label_transform=TO_TENSOR):
        
        self.driven = driven 
        self.driven_mask = driven_mask
        self.driven_recolor = driven_recolor
        self.driven_style_vector = driven_style_vector 
        self.target = target 
        self.target_mask = target_mask 
        self.target_style_vector = target_style_vector 
         
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.driven)

    def __getitem__(self, index):
        target_image = Image.open(self.target[index]).convert('RGB')
        driven_image = Image.open(self.driven[index]).convert('RGB')
        driven_recolor_image = Image.open(self.driven_recolor[index]).convert('RGB')
        if self.img_transform is not None:
            target_tensor = self.img_transform(target_image)
            driven_tensor = self.img_transform(driven_image)
            driven_recolor_tensor = self.img_transform(driven_image)
        
        driven_m = Image.open(self.driven_mask[index]).convert('L')
        target_m = Image.open(self.target_mask[index]).convert('L')
        if self.label_transform is not None:
            driven_m = self.label_transform(driven_m)
            target_m = self.label_transform(target_m)
            
        driven_s_v = torch.load(self.driven_style_vector[index])  #  [1,12,#style_vector], 已经带了batch_size = 1 这个维度
        target_s_v = torch.load(self.target_style_vector[index])
        
        return (driven_tensor, driven_m, driven_s_v, target_tensor, target_m, target_s_v,
                driven_recolor_image, driven_image, target_image)

# 视频换脸 stiching 时需要的 dataset
class VideoFaceSwappingStichingDataset(Dataset):
    def __init__(self, swapped_mask, swapped_style_vector, content_img, border_img,
                 img_transform=TO_TENSOR, label_transform=TO_TENSOR):
        
        self.swapped_mask = swapped_mask 
        self.swapped_style_vector = swapped_style_vector
        
        self.content_img = content_img  # 提供脸部约束的图片, 一般是PTI finetune之后的模型，在交换mask后重建的图片
        self.border_img = border_img  # 提供boundary处约束的图片， 一般是 target 视频
          
        self.img_transform = img_transform
        self.label_transform = label_transform
        
        
    def __len__(self):
        return len(self.swapped_mask)

    def __getitem__(self, index):
        content_image = Image.open(self.content_img[index]).convert('RGB')
        border_image = Image.open(self.border_img[index]).convert('RGB')
        if self.img_transform is not None:
            content_image = self.img_transform(content_image)
            border_image = self.img_transform(border_image)
        
        swapped_m = Image.open(self.swapped_mask[index]).convert('L')
        if self.label_transform is not None:
            swapped_m = self.label_transform(swapped_m)
            
        swapped_s_v_np = np.load(self.swapped_style_vector[index])  #  [1,12,#style_vector], 已经带了batch_size = 1 这个维度
        swapped_s_v = torch.from_numpy(swapped_s_v_np)
        
        
        return content_image, border_image, swapped_m, swapped_s_v
            