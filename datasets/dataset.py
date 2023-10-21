import os.path as osp
from typing import Union
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import random
import torch
import torchvision.transforms.functional as TF

from datasets.utils import make_dataset


# 18个属性,skin-1,nose-2,...cloth-18
# 额外的是0,表示背景
celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']

# face-parsing.PyTorch 虽然也用的是 CelebA-Mask-HQ 中的19个属性 ，但属性的顺序不太一样
FFHQ_label_list = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']  # skin-1 l_brow-2 ...
 
# 9个属性
faceParser_label_list = ['background', 'mouth', 'eyebrows',
                         'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface']

# 12个属性
faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 'nose', 'skin', 'ears', 'belowface','mouth','eye_glass','ear_rings']

TO_TENSOR = transforms.ToTensor()
MASK_CONVERT_TF = transforms.Lambda(
    lambda celebAHQ_mask: __celebAHQ_masks_to_faceParser_mask(celebAHQ_mask))

MASK_CONVERT_TF_DETAILED = transforms.Lambda(
    lambda celebAHQ_mask: __celebAHQ_masks_to_faceParser_mask_detailed(celebAHQ_mask))

FFHQ_MASK_CONVERT_TF = transforms.Lambda(
    lambda mask: __ffhq_masks_to_faceParser_mask(mask))

FFHQ_MASK_CONVERT_TF_DETAILED = transforms.Lambda(
    lambda mask: __ffhq_masks_to_faceParser_mask_detailed(mask))

NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def get_transforms(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __ffhq_masks_to_faceParser_mask_detailed(mask):
    """将 ffhq 的mask (用face-parsing.PyTorch模型提取到的) 转为 faceParser格式（聚合某些 facial component）

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        返回转化后的mask，同样是shape [H,W]，但是类别数更少
    """

    # assert len(mask.size) == 2, "mask 必须是[H,W]格式的数据"

    converted_mask = np.zeros_like(mask)

    backgorund = np.equal(mask, 0)
    converted_mask[backgorund] = 0

    lip = np.logical_or(np.equal(mask, 12), np.equal(mask, 13))  # 嘴唇
    converted_mask[lip] = 1

    eyebrows = np.logical_or(np.equal(mask, 2),
                             np.equal(mask, 3))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(mask, 4), np.equal(mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(mask, 17)
    converted_mask[hair] = 4

    nose = np.equal(mask, 10)
    converted_mask[nose] = 5

    skin = np.equal(mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(mask, 7), np.equal(mask, 8))
    converted_mask[ears] = 7

    belowface = np.equal(mask, 14)
    converted_mask[belowface] = 8
    
    mouth = np.equal(mask, 11)   # 牙齿
    converted_mask[mouth] = 9

    eye_glass = np.equal(mask, 6)   # 眼镜
    converted_mask[eye_glass] = 10
    
    ear_rings = np.equal(mask, 9)   # 耳环
    converted_mask[ear_rings] = 11

    return converted_mask

def __ffhq_masks_to_faceParser_mask(mask):
    """将 ffhq 的mask (用face-parsing.PyTorch模型提取到的) 转为 faceParser格式（聚合某些 facial component）

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        返回转化后的mask，同样是shape [H,W]，但是类别数更少
    """

    assert len(mask.size) == 2, "mask 必须是[H,W]格式的数据"

    converted_mask = np.zeros_like(mask)

    backgorund = np.equal(mask, 0)
    converted_mask[backgorund] = 0

    mouth = np.logical_or(
        np.logical_or(np.equal(mask, 11), np.equal(mask, 12)),
        np.equal(mask, 13)
    )
    converted_mask[mouth] = 1

    eyebrows = np.logical_or(np.equal(mask, 2),
                             np.equal(mask, 3))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(mask, 4), np.equal(mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(mask, 17)
    converted_mask[hair] = 4

    nose = np.equal(mask, 10)
    converted_mask[nose] = 5

    skin = np.equal(mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(mask, 7), np.equal(mask, 8))
    converted_mask[ears] = 7

    belowface = np.equal(mask, 14)
    converted_mask[belowface] = 8

    return converted_mask

def __celebAHQ_masks_to_faceParser_mask_detailed(celebA_mask):
    """将 celebAHQ_mask 格式的mask 转为 faceParser格式（聚合某些 facial component）
    
    保持一些细节，例如牙齿

    Args:
        celebA_mask (PIL image): with shape [H,W]
    Return:
        返回转化后的mask，同样是shape [H,W]，但是类别数更少
    """

    assert len(celebA_mask.size) == 2, "mask 必须是[H,W]格式的数据"

    converted_mask = np.zeros_like(celebA_mask)

    backgorund = np.equal(celebA_mask, 0)
    converted_mask[backgorund] = 0

    lip = np.logical_or(np.equal(celebA_mask, 11), np.equal(celebA_mask, 12))
    converted_mask[lip] = 1

    eyebrows = np.logical_or(np.equal(celebA_mask, 6),
                             np.equal(celebA_mask, 7))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(celebA_mask, 4), np.equal(celebA_mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(celebA_mask, 13)
    converted_mask[hair] = 4

    nose = np.equal(celebA_mask, 2)
    converted_mask[nose] = 5

    skin = np.equal(celebA_mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(celebA_mask, 8), np.equal(celebA_mask, 9))
    converted_mask[ears] = 7

    belowface = np.equal(celebA_mask, 17)
    converted_mask[belowface] = 8
    
    mouth = np.equal(celebA_mask, 10)   # 牙齿
    converted_mask[mouth] = 9

    eye_glass = np.equal(celebA_mask, 3)   # 眼镜
    converted_mask[eye_glass] = 10
    
    ear_rings = np.equal(celebA_mask, 15)   # 耳环
    converted_mask[ear_rings] = 11
    
    # r_ear = np.equal(celebA_mask, 9)  # 右耳
    # converted_mask[r_ear] = 12
    
    return converted_mask

def __celebAHQ_masks_to_faceParser_mask(celebA_mask):
    """将 celebAHQ_mask 格式的mask 转为 faceParser格式（聚合某些 facial component）

    Args:
        celebA_mask (PIL image): with shape [H,W]
    Return:
        返回转化后的mask，同样是shape [H,W]，但是类别数更少
    """

    assert len(celebA_mask.size) == 2, "mask 必须是[H,W]格式的数据"

    converted_mask = np.zeros_like(celebA_mask)

    backgorund = np.equal(celebA_mask, 0)
    converted_mask[backgorund] = 0

    mouth = np.logical_or(
        np.logical_or(np.equal(celebA_mask, 10), np.equal(celebA_mask, 11)),
        np.equal(celebA_mask, 12)
    )
    converted_mask[mouth] = 1

    eyebrows = np.logical_or(np.equal(celebA_mask, 6),
                             np.equal(celebA_mask, 7))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(celebA_mask, 4), np.equal(celebA_mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(celebA_mask, 13)
    converted_mask[hair] = 4

    nose = np.equal(celebA_mask, 2)
    converted_mask[nose] = 5

    skin = np.equal(celebA_mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(celebA_mask, 8), np.equal(celebA_mask, 9))
    converted_mask[ears] = 7

    belowface = np.equal(celebA_mask, 17)
    converted_mask[belowface] = 8

    return converted_mask


class CelebAHQDataset(Dataset):
    """
    CelebA-HQ数据集，具体数据来自于 https://github.com/ZPdesu/SEAN
    """
    def __init__(self, dataset_root, mode="test",
                 img_transform=TO_TENSOR, label_transform=TO_TENSOR,
                 load_vis_img=False, fraction=1.0,
                 flip_p=-1,  # negative means not flipping
                 specific_ids: Union[list, tuple] = None,
                 paired: bool = False,
                 shuffle: bool = False,
                 ):
        assert mode in ("train", "test", "all"), "CelebAHQDataset mode type unsupported!"
        self.mode = mode
        if mode in ("all",):
            self.roots = [osp.join(dataset_root, "train"), osp.join(dataset_root, "test")]
        else:
            self.roots = [osp.join(dataset_root, self.mode)]
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.load_vis_img = load_vis_img
        self.fraction = fraction
        self.flip_p = flip_p
        self.paired = paired

        self.imgs = []
        self.labels = []
        self.labels_vis = []
        for root in self.roots:
            imgs = sorted(make_dataset(osp.join(root, "images")))
            imgs = imgs[:int(len(imgs)*self.fraction)]

            labels = sorted(make_dataset(osp.join(root, "labels")))
            labels = labels[:int(len(labels)*self.fraction)]

            labels_vis = sorted(make_dataset(osp.join(root, "vis"))) if self.load_vis_img else None
            labels_vis = labels_vis[:int(len(labels_vis)*self.fraction)] if self.load_vis_img else []

            self.imgs.extend(imgs)
            self.labels.extend(labels)
            self.labels_vis.extend(labels_vis)

        self.imgs, self.labels, self.labels_vis = self._filter_specific_ids(specific_ids)

        if self.load_vis_img:
            assert len(self.imgs) == len(self.labels) == len(self.labels_vis)
        else:
            assert len(self.imgs) == len(self.labels)

        print(f"[CelebAHQDataset] files loaded. mode={self.mode}, #imgs={len(self.imgs)}, "
              f"#labels={len(self.labels)}, #vis={len(self.labels_vis)}")

        # # 优化 600 个iteration 的style code保存路径
        # self.optim_codes_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/v0_8_stage2_entypeSEAN/optim_Results"
        
        # image pairs indices
        self.indices = np.arange(len(self.imgs))

        # TODO: shuffle the indices
        if shuffle:
            np.random.shuffle(self.indices)

        self.pair_indices = self.indices.reshape(-1, 2)

    def __len__(self):
        if not self.paired:
            return len(self.indices)
        else:
            return len(self.pair_indices)

    def _filter_specific_ids(self, specific_ids: tuple):
        """ filter the images according to the specific_ids
        """
        if specific_ids is None:
            return self.imgs, self.labels, self.labels_vis
        elif self.fraction < 1.0:
            raise ValueError("[CelebAHQDataset] specific_ids and fraction cannot be set simultaneously!")

        # parse the tuple into two lists, e.g. (("train","12"), ("test","45")) -> ("train","train") and ("12","45")
        spec_modes, spec_ids = [], []
        id_order_dict = {}
        for idx, spec_id in enumerate(specific_ids):
            one_mode, one_id = spec_id[0], spec_id[1]
            spec_modes.append(one_mode)
            spec_ids.append(one_id)
            id_order_dict[one_id] = {
                "mode": one_mode, "order": idx,
            }

        # filter and re-order
        ret_imgs = [""] * len(specific_ids)
        ret_labels = [""] * len(specific_ids)
        ret_labels_vis = [""] * len(specific_ids)
        found_cnt = 0
        for k in range(len(spec_ids)):  # target specific ids
            one_spec_mode = spec_modes[k]
            one_spec_id = spec_ids[k]
            for idx in range(len(self.imgs)):  # full dataset
                one_img = self.imgs[idx]
                one_label = self.labels[idx]
                one_label_vis = self.labels_vis[idx] if self.load_vis_img else None
                if one_spec_mode in one_img and one_spec_id == osp.basename(one_img):  # found one
                    found_cnt += 1
                    one_spec_order = id_order_dict[one_spec_id]["order"]
                    ret_imgs[one_spec_order] = one_img
                    ret_labels[one_spec_order] = one_label
                    ret_labels_vis[one_spec_order] = one_label_vis
                    break

        if found_cnt < len(specific_ids):
            print(f"[[Warning]][CelebAHQDataset] not enough images found (={found_cnt}) for "
                  f"specific ids (={len(specific_ids)})!")

        ret_imgs = list(filter(None, ret_imgs))
        ret_labels = list(filter(None, ret_labels))
        ret_labels_vis = list(filter(None, ret_labels_vis))
        return ret_imgs, ret_labels, ret_labels_vis

    def load_single_image(self, index):
        """把一张图片的 原图, seg mask, 以及mask对应可视化的图都加载进来
        Args:
            index (int): 图片的索引
        Return:
            img: RGB图
            label: seg mask
            label_vis: seg mask的可视化图
        """
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)

        label = self.labels[index]
        # label = osp.join("/apdcephfs/share_1290939/zhianliu/py_projects/our_editing/ui_results","%s_mask.png"%osp.basename(label)[:-4])
        label = Image.open(label).convert('L')
        if self.label_transform is not None:
            label = self.label_transform(label)

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        return img, label, label_vis, img_path

    def _output_item(self, idx):
        if not self.paired:
            index = self.indices[idx]
            img, label, label_vis, img_path = self.load_single_image(index)
            if self.flip_p > 0:
                if random.random() < self.flip_p:
                    img = TF.hflip(img)
                    label = TF.hflip(label)
            return img, label, label_vis, img_path
        else:
            index1 = self.indices[idx * 2]
            index2 = self.indices[idx * 2 + 1]
            img1, label1, label_vis1, img_path1 = self.load_single_image(index1)
            img2, label2, label_vis2, img_path2 = self.load_single_image(index2)
            if self.flip_p > 0:
                if random.random() < self.flip_p:
                    img1 = TF.hflip(img1)
                    label1 = TF.hflip(label1)
                if random.random() < self.flip_p:
                    img2 = TF.hflip(img2)
                    label2 = TF.hflip(label2)
            return {
                "bag1": (img1, label1, label_vis1, img_path1),
                "bag2": (img2, label2, label_vis2, img_path2)
            }

    def __getitem__(self, idx):
        return self._output_item(idx)
    
        # # 1阶段重建的图片
        # img_name = osp.basename(self.imgs[index])[:-4]
        # recon_img = Image.open(osp.join(self.optim_codes_dir,img_name,"%s_recon.png"%img_name)).convert('RGB')
        # if self.img_transform is not None:
        #     recon_img = self.img_transform(recon_img)
            
        # # 优化后的code
        # optim_code_path = osp.join(self.optim_codes_dir,img_name,"%s_0600.npy"%img_name)
        # assert osp.exists(optim_code_path), "%s 文件不存在!"%optim_code_path
        # optimed_style_code = np.load(optim_code_path)[0]
        
        # return img, recon_img, optimed_style_code, label, label_vis
        
        # pair_indices = self.pair_indices[idx, :]

        # img1, label1, label_vis1 = self.load_single_image(pair_indices[0])
        # img2, label2, label_vis2 = self.load_single_image(pair_indices[1])

        # return (img1, img2), (label1, label2), (label_vis1, label_vis2)


class FolderDataset(Dataset):
    """
    CelebA-HQ数据集，具体数据来自于 https://github.com/ZPdesu/SEAN
    """
    def __init__(self, dataset_root, mode="",
                 img_transform=TO_TENSOR, label_transform=TO_TENSOR,
                 load_vis_img=False, fraction=1.0,
                 flip_p=-1):  # 负数表示不翻转
        self.mode = mode
        self.root = osp.join(dataset_root, self.mode)
        self.img_transform = transforms.Compose(
            [transforms.Resize((512, 512)), img_transform]
        )
        self.fraction = fraction
        self.flip_p = flip_p

        self.imgs = sorted(make_dataset(osp.join(self.root, "")))
        self.imgs = self.imgs[:int(len(self.imgs) * self.fraction)]

        # image pairs indices
        self.indices = np.arange(len(self.imgs))
        self.pair_indices = self.indices.reshape(-1, 2)

    def __len__(self):
        return len(self.indices)

    def load_single_image(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, img_path

    def __getitem__(self, idx):
        index = self.indices[idx]

        img, path = self.load_single_image(index)

        if self.flip_p > 0:
            if random.random() < self.flip_p:
                img = TF.hflip(img)

        return img, path


class FFHQDataset(Dataset):
    """
    FFHQ数据集，提取 mask 的方式参照了Babershop，用的是BiSegNet提取的
    """

    def __init__(self, dataset_root,
                 img_transform=TO_TENSOR, label_transform=TO_TENSOR,
                 fraction=1.0,
                 load_raw_label=False,
                 flip_p = -1):

        self.root = dataset_root
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.fraction=fraction
        self.load_raw_label = load_raw_label
        self.flip_p = flip_p
        
        with open(osp.join(self.root,"images_1024","ffhq_list.txt"),"r") as f:
            f_lines = f.readlines()
        
        self.imgs = sorted([osp.join(self.root, "images_1024", line.replace("\n","")) for line in f_lines])
        self.imgs = self.imgs[:int(len(self.imgs)*self.fraction)]
        self.labels = [img.replace("images_1024","BiSeNet_mask") for img in self.imgs]
    
        assert len(self.imgs) == len(self.labels)
        
        self.indices = np.arange(len(self.imgs))

    def __len__(self):
        return len(self.indices)

    def load_single_image(self, index):
        """把一张图片的 原图, seg mask, 以及mask对应可视化的图都加载进来

        Args:
            index (int): 图片的索引
        Return:
            img: RGB图
            label: seg mask
            label_vis: seg mask的可视化图
        """
        img = self.imgs[index]
        img = Image.open(img).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)

        label = self.labels[index]
        label = Image.open(label).convert('L')
        
        if self.load_raw_label:
            original_label = TO_TENSOR(label)
        
        if self.label_transform is not None:
            label = self.label_transform(label)

        label_vis = -1  # unified interface
        
        if self.load_raw_label:
            return img, original_label, label, label_vis
        else:
            return img, label, label_vis
        
    def __getitem__(self, idx):
        index = self.indices[idx]

        img, label, label_vis = self.load_single_image(index)
        
        if self.flip_p > 0:
            if random.random() < self.flip_p:
                img = TF.hflip(img)
                label = TF.hflip(label)
        
        return img, label, label_vis    


# 视频换脸，第一阶段（包括驱动+手动编辑mask+交换style vectors）换脸后的数据集
class VideoFaceSwappingStageOneDataset(Dataset):

    def __init__(self, target_images, driven_images, UI_edit_masks, swapped_style_vectors,
                 img_transform=TO_TENSOR, label_transform=TO_TENSOR):
        
        self.target_images = target_images # target 视频的所以帧
        self.driven_images = driven_images # faceVid2Vid + GPEN 的驱动的图片
        self.UI_edit_masks = UI_edit_masks # 手动编辑后的mask
        self.swapped_style_vectors =  swapped_style_vectors # 第一阶段交换style vectors之后的结果
        assert len(self.target_images) == len(self.driven_images) == len(self.UI_edit_masks) == len(self.swapped_style_vectors), "所有视频的长度应该一样"
        
        self.img_transform = img_transform
        self.label_transform = label_transform
        
        
    def __len__(self):
        return len(self.target_images)

    def __getitem__(self, index):
        target_image = Image.open(self.target_images[index]).convert('RGB')
        driven_image = Image.open(self.driven_images[index]).convert('RGB')
        if self.img_transform is not None:
            target_image = self.img_transform(target_image)
            driven_image = self.img_transform(driven_image)
        
        edit_mask = Image.open(self.UI_edit_masks[index]).convert('L')
        if self.label_transform is not None:
            edit_mask = self.label_transform(edit_mask)
            
        style_vector = torch.load(self.swapped_style_vectors[index])  #  [1,12,#style_vector], 已经带了batch_size = 1 这个维度


        return target_image, driven_image, edit_mask, style_vector


if __name__ == "__main__":
    celeba_hq_folder = "/home/yuange/datasets/CelebA-HQ/"
    specific_mode_imgs = [
        ("test", "28063.jpg"),
        ("test", "28426.jpg"),
        ("test", "28297.jpg"),
        ("test", "28629.jpg"),
        ("train", "4829.jpg"),
        ("train", "5982.jpg"),
        ("train", "4612.jpg"),
        ("train", "4811.jpg"),
        ("test", "29404.jpg"),
        ("test", "29386.jpg"),
        ("test", "28740.jpg"),
        ("test", "28393.jpg"),
        ("test", "28072.jpg"),
        ("test", "29318.jpg"),
        ("test", "29989.jpg"),
        ("test", "28835.jpg"),
        ("train", "724.jpg"),
        ("train", "1123.jpg"),
        ("test", "29756.jpg"),
        ("test", "29220.jpg"),
        ("test", "28021.jpg"),
        ("test", "29833.jpg")
    ]
    dataset_celebahq = CelebAHQDataset(
        celeba_hq_folder,
        mode="all",
        specific_ids=specific_mode_imgs,
        load_vis_img=True,
        paired=False,

    )
