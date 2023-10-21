from PIL import Image
import torch
import numpy as np
import glob
import os

from face_swap_for_video import faceSwapping_pipeline
from options.our_swap_face_pipeline_options import OurSwapFacePipelineOptions
from models.networks import Net3
from utils import torch_utils





source = '/apdcephfs/share_1290939/zhianliu/datasets/video_swapping_demo/celebrate/zelensky.jpg'
target_path = '/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video_res/swap_zelensky_to_target1/crop/'

n = 200

target_frames = sorted(glob.glob(target_path + '*.png'))

source_name = os.path.basename(source).split('.')[0]
# target_name = target_path.split('/')[-3]
target_name = '056'

# swap_' + source_name + '_to_' + target_name # + '_optim_w'
save_name = 'swap_zelensky_to_target1'
save_dir = '/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video_res/' + save_name
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


opts = OurSwapFacePipelineOptions().parse()
# opts.PTI_checkpoint_path = '/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/v_18_video_swapping/28494_to_874/finetuned_G_lr0.001000_iters150.pth'

net = Net3(opts)
net = net.to(opts.device)
# save_dict = torch.load(opts.PTI_checkpoint_path)
save_dict = torch.load(opts.checkpoint_path)
net.load_state_dict(torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module."))
net.latent_avg = save_dict['latent_avg'].to(opts.device)
print("Load LocalStyleGAN pre-trained model success!")


for target in target_frames[:n]:
    name = os.path.basename(target).split('.')[0]

    res = faceSwapping_pipeline(source=source, 
                                target=target, 
                                opts=opts, 
                                net=net,
                                save_dir=save_dir, 
                                target_mask=None, 
                                need_crop =False, 
                                verbose=False, 
                                only_target_crop=False,
                                only_source_crop=True,
                                optimize_W=False,
                                finetune_net=False,
                                copy_face=False,
                                pose_drive='faceVid2Vid',
                                face_enhancement='gpen',
                                name=name)
    
    # res.save(os.path.join(save_dir, name + '.png'))