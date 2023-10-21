
import os.path as osp
from PIL import Image
from tqdm import tqdm
import numpy as np

gt_base_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/images"
base_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/ablation_study/"

exp_dirs = [
    "v_15_exp3_seg12_finetuneGD_8A100_remainLyrIdx11_flip_celeba_200KIters",
    "v_15_baseline_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_reRun2",
    "v_15_exp1_seg12_finetuneGD_8A100_remainLyrIdx15_flip_celeba_200KIters",
    "v_15_exp2_seg12_finetuneGD_8A100_remainLyrIdx17_flip_celeba_200KIters"  
]

save_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/ablation_study/different_layers_recon"

size = 256
for i in tqdm(range(2000), total=2000):
    gt = np.array(Image.open(osp.join(gt_base_dir,"%05d.jpg"%(i+28000))).resize((size, size)))
    lyr_11 = np.array(Image.open(osp.join(base_dir,exp_dirs[0], "test_recon_200000", "%05d_recon_face.png"%(i+28000))).resize((size, size)))
    lyr_13 = np.array(Image.open(osp.join(base_dir,exp_dirs[1], "test_recon_200000", "%05d_recon_face.png"%(i+28000))).resize((size, size)))
    lyr_15 = np.array(Image.open(osp.join(base_dir,exp_dirs[2], "test_recon_200000", "%05d_recon_face.png"%(i+28000))).resize((size, size)))
    lyr_17 = np.array(Image.open(osp.join(base_dir,exp_dirs[3], "test_recon_200000", "%05d_recon_face.png"%(i+28000))).resize((size, size)))
    
    
    big_pic = Image.fromarray(np.hstack([gt, lyr_11, lyr_13, lyr_15, lyr_17]))
    big_pic.save(osp.join(save_dir, "%05d.png"%(i+28000)))
    
    