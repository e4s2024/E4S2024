import imp
import os.path as osp
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
import pandas as pd
import shutil

def mv_imgs():
    recon_base_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/v_15_hybrid_stage1_seg12_finetuneGD_8A100_pspHyperParas_remainLyrIdx13_flip_200KIters/optim_Results_120000_lr1e2"

    for img_name in tqdm(range(2000)):
        shutil.copyfile(
            osp.join(recon_base_dir,"%05d"%(img_name+28000),"%05d_0050.png"%(img_name+28000)),
            osp.join("/apdcephfs/share_1290939/zhianliu/pretrained_models/sofGAN/tmp","%05d_0050.png"%(img_name+28000)),
        )

def calculate_metrics(size=1024):
    
    print(size)
    
    img_names, SSIMs, PSNRs, RMSEs = [],[],[],[]
    
    for img_name in tqdm(range(2000)):
        
        # if img_name == 813:
        #     continue
        
        # recon_base_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/Mask_Guided_Portrait_Editing/recon_results"
        # # recon_base_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/CelebAMask-HQ/recon_results"
        # recon_base_dir = "/apdcephfs/share_1290939/zhianliu/running_results/SEAN/CelebA-HQ_pretrained/test_latest/images/synthesized_image"
        # recon_base_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/SPADE/recon_results"
        # recon = np.asarray(Image.open(osp.join(recon_base_dir,"%05d.png"%(img_name+28000))).resize((size,size)))
        
        # recon_base_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/v_15_hybrid_stage1_seg12_finetuneGD_8A100_pspHyperParas/"
        # recon_base_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/v_15_hybrid_stage1_seg12_finetuneGD_8A100_pspHyperParas_remainLyrIdx13_flip_200KIters/"
        # recon_base_dir = "/apdcephfs/share_1290939/zhianliu/pretrained_models/sofGAN/optim_res_dir_iters1K"
        recon_base_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/ablation_study/v_15_exp8_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_noMsEncoder/test_recon_200000"
        recon = np.asarray(Image.open(osp.join(recon_base_dir,"%05d_recon_face.png"%(img_name+28000))).resize((size, size)))

        gt_base_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/images"
        gt = np.asarray(Image.open(osp.join(gt_base_dir,"%05d.jpg"%(img_name+28000))).resize((size, size)))
        
        img_names.append("%05d_recon_face.jpg"%(img_name+28000))

        # 按照 SEAN repo 中的 issue #5 方式进行计算SSIM
        ssim = compare_ssim(gt/255.0,recon/255.0, multichannel=True, gaussian_weights=True, use_sample_covariance=False) 
        SSIMs.append(ssim)
    
        psnr = compare_psnr(gt,recon, data_range=255)
        PSNRs.append(psnr)
    
        mse = compare_mse(gt/255.0, recon/ 255.0)
        rmse = np.sqrt(mse)
        RMSEs.append(rmse)
    
    dict = {'img': img_names, 'SSIM': SSIMs, 'PSNR': PSNRs, 'RMSE': RMSEs} 
    df = pd.DataFrame(dict)
    df.to_csv(osp.join(recon_base_dir,"%d.csv"%size),index=False,sep='\t')
    
    print("SSIM:", np.mean(SSIMs))
    print("PSNR:", np.mean(PSNRs))
    print("RMSE:", np.mean(RMSEs))
    

def metric_test(metric_opt="ssim",size=1024):
    
    # gt = np.asarray(Image.open("/apdcephfs/share_1290939/zhianliu/py_projects/our_editing/metric/29057.jpg").resize((512, 512)))
    # recon = np.asarray(Image.open("/apdcephfs/share_1290939/zhianliu/py_projects/our_editing/metric/29057recon.png").resize((512, 512)))
    
    Metrics = []
    for img_name in ["28006","28022","28031","28092","28101","28221","28298","28314","28352","28363","28380","28381","28394","28541","28905","29057","29258"]:
        recon = np.asarray(Image.open("/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/"
                                "v_14_hybridStage1_seg12_fixedGD_8V100_pspHyperParas/optim_Results_80000_lr1e2/%s/%s_0050.png"%(img_name,img_name)).resize((size, size)))

        gt = np.asarray(Image.open("/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/"
                                "v_14_hybridStage1_seg12_fixedGD_8V100_pspHyperParas/optim_Results_80000_lr1e2/%s/%s_gt.png"%(img_name,img_name)).resize((size, size)))
        
        if metric_opt == "ssim":
            # 注意，在算SSIM的时候，对于RGB图要指定 multichannel=True，并且范围为 255
            ssim = compare_ssim(gt,recon, multichannel=True, gaussian_weights=True, sigma=1.5, data_range=255)
            Metrics.append(ssim)
        elif metric_opt == "psnr":
            psnr = compare_psnr(gt,recon, data_range=255)
            Metrics.append(psnr)
        elif metric_opt == "rmse":
            mse = compare_mse(gt/ 255.0,recon/ 255.0)
            rmse = np.sqrt(mse)
            Metrics.append(rmse)
        else:
            raise RuntimeError("只支持 ssim, psnr, rmse!")
        
    Metrics = np.array(Metrics)
    
    return np.mean(Metrics)

# ========================== SSIM ====================================
def ssim(gt_imgs_path, imgs_path, size=1024):
    assert len(gt_imgs_path) == len(imgs_path)
    
    SSIMs = []
    for gt_img_path, img_path in tqdm(zip(gt_imgs_path, imgs_path)):
        gt = np.asarray(Image.open(gt_img_path).resize((size, size)))
        img = np.asarray(Image.open(img_path).resize((size, size)))
        
        # 注意，在算SSIM的时候，对于RGB图要指定 multichannel=True，并且范围为 255
        ssim = compare_ssim(gt,img, multichannel=True, gaussian_weights=True, sigma=1.5, data_range=255)

        SSIMs.append(ssim)
    
    SSIMs = np.array(SSIMs)
    
    return np.mean(SSIMs)


# ========================== PSNR ====================================    
def psnr_test():
    
    gt = np.asarray(Image.open("/apdcephfs/share_1290939/zhianliu/py_projects/our_editing/metric/29057.jpg").resize((512, 512)))
    recon = np.asarray(Image.open("/apdcephfs/share_1290939/zhianliu/py_projects/our_editing/metric/29057recon.png").resize((512, 512)))

    psnr = compare_psnr(gt,recon, data_range=255)

    print(psnr)

def psnr(gt_imgs_path, imgs_path, size=1024):
    assert len(gt_imgs_path) == len(imgs_path)
    
    PSNRs = []
    for gt_img_path, img_path in tqdm(zip(gt_imgs_path, imgs_path)):
        gt = np.asarray(Image.open(gt_img_path).resize((size, size)))
        img = np.asarray(Image.open(img_path).resize((size, size)))
        
        psnr = compare_psnr(gt,img, data_range=255)
        PSNRs.append(psnr)
    
    PSNRs = np.array(PSNRs)
    
    return np.mean(PSNRs)


# ========================== RMSE ====================================
def rmse_test():
    
    gt = np.asarray(Image.open("/apdcephfs/share_1290939/zhianliu/py_projects/our_editing/metric/29057.jpg").resize((512, 512))) /255
    recon = np.asarray(Image.open("/apdcephfs/share_1290939/zhianliu/py_projects/our_editing/metric/29057recon.png").resize((512, 512))) /255
    
    mse = compare_mse(gt,recon)
    rmse = np.sqrt(mse)
    
    print(rmse)
    
def rmse(gt_imgs_path, imgs_path, size=1024):
    assert len(gt_imgs_path) == len(imgs_path) 
    
    RMSEs = []
    for gt_img_path, img_path in tqdm(zip(gt_imgs_path, imgs_path)):
        gt = np.asarray(Image.open(gt_img_path).resize((size, size))) / 255.0
        img = np.asarray(Image.open(img_path).resize((size, size))) / 255.0
        
        rmse = np.sqrt(compare_mse(gt,img))
        RMSEs.append(rmse)
    
    RMSEs = np.mean(RMSEs)
    
    return np.array(RMSEs)


# gt_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/v_14_hybridStage1_seg12_fixedGD_8V100_pspHyperParas/test_imgs"
# recon_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/v_14_hybridStage1_seg12_fixedGD_8V100_pspHyperParas/test_recon_80000"
# gt_imgs_path=sorted(glob.glob(osp.join(gt_dir,"*_input_face.png")))
# imgs_path=sorted(glob.glob(osp.join(recon_dir,"*_recon_face.png")))
# # res = ssim(gt_imgs_path, imgs_path)

# for metric_opt in ["ssim","psnr","rmse"]:
#     for size in [1024,512,256]:
#         res = metric_test(metric_opt=metric_opt,size=size)
    
#         print(metric_opt, size, res)


# mv_imgs()
calculate_metrics()
