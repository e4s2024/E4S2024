import os
import os.path as osp
from PIL import Image
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA
import glob
import cv2

def show_hole_map():
    img = Image.open("./tmp/swap_28285_to_28018.png").convert("RGB")
    hole_map = Image.open("./tmp/hole_map.png").convert("L").resize((img.height, img.width))
    img_arr = np.array(img)
    hole_arr = np.array(hole_map)

    hole_regions = np.repeat((hole_arr == 255)[:, :, np.newaxis], 3, axis=2)

    img_arr[hole_regions] = 255

    print(-1)


def ablation_recon_compare():  # ablation study 的可视化比较
    gt_base_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/images"
    recon_base_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/ablation_study/"
    save_dir = "/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/ablation_study/recon_comp"
    
    exp_names = [
        "v_15_baseline_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_reRun2",
        "v_15_exp4_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_noLPIPS",
        "v_15_exp5_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_noIDLoss",
        "v_15_exp6_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_noFaceParsingLoss",
        "v_15_exp7_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_noAdvLoss",
        "v_15_exp8_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_noMsEncoder"
    ]
    
    for i in tqdm(range(2000), total=2000):
        gt = np.asarray(Image.open(osp.join(gt_base_dir,"%05d.jpg"%(i+28000))).resize((256, 256)))
        recons = [gt]
        for j, exp_name in enumerate(exp_names):
            recon_j = Image.open(osp.join(recon_base_dir,exp_name,"test_recon_200000","%05d_recon_face.png"%(i+28000))).convert("RGB").resize((256, 256))
            recons.append(np.array(recon_j))
        recons.append(np.zeros((256,256,3)).astype(np.uint8))
        

        big_pic = []
        num_row = len(recons)//4
        for r in range(num_row):
            row = np.hstack(recons[r*4:(r+1)*4])
            big_pic.append(row)
        
        big_pic = np.vstack(big_pic)
        Image.fromarray(big_pic).save(os.path.join(save_dir, "%05d.png"%(i+28000)))


def comp_swap_face_with_SOTA():  # 和SOTA比较换脸的结果
    ds_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ-swap"
    with open(os.path.join(ds_dir, "pairs.txt"),"r") as f:
        lines = f.readlines()
        paris = [l.strip().split("\t") for l in lines]

    comp_save_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ-swap/SOTA_comp"
    SOTA_methods = ["FSGAN", "SimSwap", "faceshifter", "hififace", "MegaFS", "ours"]
    for s, t in tqdm(paris, total=len(paris)):
        # if os.path.exists(os.path.join(comp_save_dir, "swap_%s_to_%s.png"%(s,t))): continue
        if (s == "29207" and  t == "29066") or (s == "29944" and  t == "28813"):  # FSGAN 跑这个pair报错，因此跳过
            continue
        
        source = os.path.join(ds_dir, "images", "%s.jpg"%s)
        source_img = Image.open(source).convert("RGB").resize((256,256))
        target = os.path.join(ds_dir, "images", "%s.jpg"%t)
        target_img = Image.open(target).convert("RGB").resize((256,256))
        
        big_pic = [np.array(source_img), np.array(target_img)]
        for method in SOTA_methods:
            if method == "FSGAN" :
                result_img = Image.open(os.path.join(ds_dir, method, "%s_%s.jpg"%(s,t))).convert("RGB").resize((256,256))
            elif method == "faceshifter":
                result_img = Image.open(os.path.join(ds_dir, "faceshifter_v2", "paste_back_swap_%s_to_%s.png"%(s,t))).convert("RGB").resize((256,256))
            elif method == "ours":
                result_img = Image.open(os.path.join(ds_dir, "ours_BgPasted_dilation5", "swap_%s_to_%s.png"%(s,t))).convert("RGB").resize((256,256))
            else:
                result_img = Image.open(os.path.join(ds_dir, method, "swap_%s_to_%s.png"%(s,t))).convert("RGB").resize((256,256))
            
            big_pic.append(np.array(result_img))
        big_pic = np.hstack(big_pic)
        
        Image.fromarray(big_pic).save(os.path.join(comp_save_dir, "swap_%s_to_%s.png"%(s,t)))
        


def comp_headPose_with_SOTA():
    ds_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ-ourSwap500"
    with open(os.path.join(ds_dir, "pairs.txt"),"r") as f:
        lines = f.readlines()
        paris = [l.strip().split("\t") for l in lines]

    target_pose_file = os.path.join(ds_dir, "images_headPose.txt")
    target_pose_dict = {}
    with open(target_pose_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            seps = line.strip().split('\t')
            frame_name = seps[0].split('.')[0]
            pose = list(map(float, seps[1].split(" ")))
            target_pose_dict["%s"%frame_name] = np.array(pose)
            
    SOTA_methods = ["FSGAN", "SimSwap", "faceshifter", "hififace", "MegaFS", "ours"]
    # SOTA_methods = ["oursMaskEdited"]
    # SOTA_methods = ["ours"]
    for method in SOTA_methods:
        if method == "faceshifter":
            pose_file = os.path.join(ds_dir, "faceshifter_headPose.txt")
        else:
            pose_file = os.path.join(ds_dir, "%s_headPose.txt"%method)

        method_pose_dict = {}
        with open(pose_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                seps = line.strip().split('\t')
                frame_name = seps[0].split(".")[0]
                pose = list(map(float, seps[1].split(" ")))
                method_pose_dict["%s"%frame_name] = np.array(pose)
        
        avg_l2_distance = []
        xyz_distance = []
        pair_name = []        
        if method == "FSGAN":
            for k, v in method_pose_dict.items():
                s, t = k.split("_")
                
                if t not in target_pose_dict: continue
                # if t in ["28269","29126","28235","29722","29256","29201"]: continue  # 眼睛不存在的情况
                l2_dist = np.sqrt(np.sum((method_pose_dict[k]-target_pose_dict[t])**2))
                avg_l2_distance.append(l2_dist)
                xyz_dist = np.sqrt((method_pose_dict[k]-target_pose_dict[t])**2)
                xyz_distance.append(xyz_dist)
                pair_name.append("%s_to_%s"%(s,t))
                
        elif method == "faceshifter":
            for k, v in method_pose_dict.items():
                if len(k.split("_")) <6: 
                    break
                s, t = k.split("_")[3], k.split("_")[5]
                
                if t not in target_pose_dict: continue
                # if t in ["28269","29126","28235","29722","29256","29201"]: continue  # 眼睛不存在的情况
                l2_dist = np.sqrt(np.sum((method_pose_dict[k]-target_pose_dict[t])**2))
                avg_l2_distance.append(l2_dist)
                xyz_dist = np.sqrt((method_pose_dict[k]-target_pose_dict[t])**2)
                xyz_distance.append(xyz_dist)
                pair_name.append("%s_to_%s"%(s,t))
        else:
            for k, v in method_pose_dict.items():
                s, t = k.split("_")[1], k.split("_")[3]
                
                if t not in target_pose_dict: continue
                # if t in ["28269","29126","28235","29722","29256","29201"]: continue  # 眼睛不存在的情况
                l2_dist = np.sqrt(np.sum((method_pose_dict[k]-target_pose_dict[t])**2))
                avg_l2_distance.append(l2_dist)   
                xyz_dist = np.sqrt((method_pose_dict[k]-target_pose_dict[t])**2)
                xyz_distance.append(xyz_dist)
                pair_name.append("%s_to_%s"%(s,t))
                
        l2_dist = np.average(avg_l2_distance)
        xyz_dist = np.average(xyz_distance,axis=0)
        print(method, l2_dist, xyz_dist)
        


def comp_expression_with_SOTA():
    ds_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ-ourSwap500"
    with open(os.path.join(ds_dir, "pairs.txt"),"r") as f:
        lines = f.readlines()
        paris = [l.strip().split("\t") for l in lines]

    target_expression_files = sorted(glob.glob(os.path.join(ds_dir, "images_3dmm", "*.txt")))
    target_expression_dict = {}
    for f in target_expression_files:
        frame_name = os.path.basename(f).split("_")[0]
        target_expression_dict[frame_name] = np.loadtxt(f).astype(np.float32)
    
    
    SOTA_methods = ["FSGAN", "SimSwap", "faceshifter", "hififace", "MegaFS", "ours"]
    for method in SOTA_methods:
        avg_l2_distance = []
    
        if method == "faceshifter":
            expression_files = sorted(glob.glob(os.path.join(ds_dir, "faceshifter_3dmm","*.txt")))
        elif method == "ours":
            expression_files = sorted(glob.glob(os.path.join(ds_dir, "ours_3dmm","*.txt")))
        else:
            expression_files = sorted(glob.glob(os.path.join(ds_dir, "%s_3dmm"%method, "*.txt")))

        method_expression_dict = {}
        for f in expression_files:
            seps = os.path.basename(f).split("_")
            if method == "FSGAN":
                s, t = seps[0], seps[1]
            elif method == "faceshifter":
                s, t = seps[3], seps[5]
            else:
                s, t = seps[1], seps[3]

            current_expression = np.loadtxt(f).astype(np.float32)
            target_expression = target_expression_dict[t]

            current_dist = np.sqrt(np.sum((current_expression - target_expression)**2))
            avg_l2_distance.append(current_dist)
    
        dist = np.average(avg_l2_distance)
        print(method, dist)        
        

def show_occlussion():
    
    driven = np.array(Image.open("./tmp/D_recon.png").convert("RGB").resize((512,512)))
    driven_mask = np.array(Image.open("./tmp/D_mask.png").convert("L"))
    # 脸部区域，也就是 除了背景、头发、belowface、耳朵的区域
    driven_hair_bg_neck_region = np.logical_or(
        np.logical_or(np.equal(driven_mask, 0), np.equal(driven_mask, 4)),
        np.equal(driven_mask, 8)
    )
    driven_ear_earings_region = np.logical_or(np.equal(driven_mask, 7), np.equal(driven_mask, 11))
    driven_non_face_region = np.logical_or(driven_hair_bg_neck_region, driven_ear_earings_region)
    driven_face_region = np.logical_not(driven_non_face_region)
    driven_face = driven_face_region[:,:,None] * driven
    Image.fromarray(driven_face).save("./tmp/driven_face.png")
    
    
    target = np.array(Image.open("./tmp/T_recon.png").convert("RGB").resize((512,512)))
    target_mask = np.array(Image.open("./tmp/T_mask.png").convert("L"))
    # 脸部区域，也就是 除了背景、头发、belowface、耳朵的区域
    target_hair_bg_neck_region = np.logical_or(
        np.logical_or(np.equal(target_mask, 0), np.equal(target_mask, 4)),
        np.equal(target_mask, 8)
    )
    target_ear_earings_region = np.logical_or(np.equal(target_mask, 7), np.equal(target_mask, 11))
    target_non_face_region = np.logical_or(target_hair_bg_neck_region, target_ear_earings_region)
    target_face_region = np.logical_not(target_non_face_region)
    target_face = target_face_region[:,:,None] * target
    Image.fromarray(target_face).save("./tmp/target_face.png")
    
    # source 缺失的皮肤区域, driven是0，target是1的区域
    missing_area = np.logical_and(driven_face_region == False, target_face_region == True)
    Image.fromarray((255*missing_area).astype(np.uint8)).save("./tmp/missing_area.png")
    # 缺失区域 在source中是什么
    image = cv2.imread("./tmp/D_recon.png")
    mask = cv2.imread("./tmp/missing_area.png")
    image = cv2.resize(image, (512, 512))
    red = [0,0,255]
    colored_missing_area = image + red * (mask/255.) * 1
    cv2.imwrite('./tmp/colored_missing_area.png', colored_missing_area)
    
    # target 遮挡的区域, driven是1，target是0的区域
    occluded_area = np.logical_and(driven_face_region == True, target_face_region == False)
    Image.fromarray((255*occluded_area).astype(np.uint8)).save("./tmp/occluded_area.png")
    # 遮挡区域 在target 中是什么
    image = cv2.imread("/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/images/28527.jpg")
    mask = cv2.imread("./tmp/occluded_area.png")
    image = cv2.resize(image, (512, 512))
    blue = [255,0,0]
    colored_occluded_area = image + blue * (mask/255.) * 1
    cv2.imwrite('./tmp/colored_occluded_area.png', colored_occluded_area)
    
    
    
    
  
    
    
# show_occlussion()
        
comp_expression_with_SOTA()
# comp_headPose_with_SOTA()
# comp_swap_face_with_SOTA()
# ablation_recon_compare()        
        