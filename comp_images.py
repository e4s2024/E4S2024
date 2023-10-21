
import torch
import os
from PIL import Image
import numpy as np

# from datasets.dataset import __celebAHQ_masks_to_faceParser_mask_detailed
# from Face_swap_with_two_imgs import FaceSwap

def load_src_tgt_indices(src_tgt_index_file):
    
    with open(src_tgt_index_file, "r") as f:
            lines = f.readlines()[1:]

    lines = [l.strip().split(' ') for l in lines]
    lines = [list(l) for l in lines]

    src_indices = [l[0] for l in lines]
    tgt_indices = [l[1] for l in lines]

    return src_indices, tgt_indices



files = ['/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/figs_index_file/fs_comp.txt',
            '/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/figs_index_file/stylegan_fs_comp.txt',
            '/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/figs_index_file/supp_fs_comp.txt',
            '/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/figs_index_file/supp_fs_hq.txt',
            '/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/figs_index_file/teaser.txt']

src_indices = []
tgt_indices = []

for f in files:
    src, tgt = load_src_tgt_indices(f)
    src_indices = src_indices + src
    tgt_indices = tgt_indices + tgt

save_dir = "/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/comp_figs_orig"
os.makedirs(save_dir, exist_ok=True)

# face_swap = FaceSwap()

for src_idx, tgt_idx in zip(src_indices, tgt_indices):
    # Face swapping procedure
    source = '/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/images/' + src_idx + '.jpg'
    if not os.path.exists(source):
        source = '/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/train/images/' +src_idx + '.jpg'

    target = '/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/images/' + tgt_idx + '.jpg'
    if not os.path.exists(target):
        target = '/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/train/images/' + tgt_idx + '.jpg'
        tgt_mask = Image.open('/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/train/labels/' + tgt_idx + '.png').convert('L')
    else:
        tgt_mask = Image.open('/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/labels/' + tgt_idx + '.png').convert('L')
    # target_mask_seg12 = __celebAHQ_masks_to_faceParser_mask_detailed(tgt_mask)

    os.system('cp ' + source + ' ' + save_dir)
    os.system('cp ' + target + ' ' + save_dir)

    '''
    res, res_driven = face_swap.face_swap_pipeline(source=source, 
                                        target=target,
                                        save_dir=save_dir, 
                                        target_mask=target_mask_seg12, 
                                        crop_mode='', 
                                        verbose=False, 
                                        optimize_W=False,
                                        finetune_net=False,
                                        copy_face=False,
                                        pose_estimation=False,
                                        pose_drive='faceVid2Vid',
                                        enhancement_mode='gpen',
                                        ct_mode='blender')
    
    save_name = 'swap_' + src_idx + '_to_' + tgt_idx
    res.save(os.path.join(save_dir, save_name + '.png'))
    res_driven.save(os.path.join(save_dir, save_name + '_driven.png'))
    '''