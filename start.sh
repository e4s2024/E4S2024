#!/bin/bash

# 单卡训练
# CXX=g++ TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3 train.py \
# --exp_dir /apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/v_14_hybridStage1_seg12_fixedGD_8A100 \
# --batch_size 2 \
# --test_batch_size 2 \
# --learning_rate 0.0001 \
# --max_steps 120000 \
# --optim_name ranger \
# --ds_frac 1.0 \
# --fsencoder_type psp \
# --l2_lambda 1.0 \
# --lpips_lambda 0.2 \
# --structure_code_lambda 0 \
# --id_lambda 0.1 \
# --face_parsing_lambda 0.1 \
# --num_seg_cls 12 \
# --dataset_name celeba_mask

# # 多卡训练
# CXX=g++ TORCH_DISTRIBUTED_DEBUG=INFO TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3  -m torch.distributed.launch \
# --nproc_per_node=8 \
# --nnodes=1 \
# --node_rank=0 \
# --master_addr=localhost \
# --master_port=22222 \
# train.py \
# --exp_dir /apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/ablation_study/v_15_exp8_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_noMsEncoder \
# --batch_size 2 \
# --test_batch_size 2 \
# --learning_rate 0.0001 
# --max_steps 200000 \
# --optim_name adam \
# --ds_frac 1.0 \
# --fsencoder_type psp \
# --l2_lambda 1.0 \
# --lpips_lambda 0.8 \
# --structure_code_lambda 0 \
# --id_lambda 0.1 \
# --face_parsing_lambda 0.1 \
# --num_seg_cls 12 \
# --dataset_name celeba \
# --style_lambda 0 \
# --style_loss_norm 1 \
# --remaining_layer_idx 13 \
# --d_every 15 \
# --g_adv_lambda 0.01 \
# --flip_p 0.5



# # # 单卡优化
# CXX=g++ TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3 optimization.py \
# --exp_dir /apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/v_15_hybrid_stage1_seg12_finetuneGD_8A100_pspHyperParas_remainLyrIdx13_flip_FFHQ_300KIters \
# --checkpoint_path /apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/v_15_hybrid_stage1_seg12_finetuneGD_8A100_pspHyperParas_remainLyrIdx13_flip_FFHQ_300KIters/checkpoints/iteration_300000.pt \
# --output_dir /apdcephfs/share_1290939/zhianliu/py_projects/our_editing_swappingFace_video/29698_to_01/FFHQ_model_optim_200iters \
# --save_intermediate \
# --verbose \
# --ds_frac 1.0 \
# --id_lambda 0.1 \
# --face_parsing_lambda 0.1 \
# --num_seg_cls 12 \
# --dataset_name ffhq \
# --style_lambda 0 \
# --style_loss_norm 1 \
# --remaining_layer_idx 13 \
# --d_every 15 \
# --g_adv_lambda 0.01 \
# --flip_p 0.5



# # # 单卡优化


# # --begin_idx 200
# # --exp_dir /apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/v_15_hybrid_stage1_seg12_finetuneGD_8A100_pspHyperParas_remainLyrIdx13_flip_200KIters \
# # --checkpoint_path /apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/v_15_hybrid_stage1_seg12_finetuneGD_8A100_pspHyperParas_remainLyrIdx13_flip_200KIters/checkpoints/iteration_120000.pt \

# # 单卡测试
# CXX=g++ TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3 test.py \
# --exp_dir /apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/ablation_study/v_15_exp8_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_noMsEncoder \
# --checkpoint_path /apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/ablation_study/v_15_exp8_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_noMsEncoder/checkpoints/iteration_200000.pt \
# --save_dir /apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/ablation_study/v_15_exp8_seg12_finetuneGD_8A100_remainLyrIdx13_flip_celeba_200KIters_noMsEncoder/test_recon_200000


# # 单卡PTI finetune
# CXX=g++ TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3 PTI_finetune.py \
# --exp_dir /apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/v_16_dilation15_PTI_finetune_100Iters_le3e5_A100 \
# --max_pti_steps 100 \
# --pti_learning_rate 3e-5 \
# --outer_dilation 15


# # 单卡 视频换脸中的 finetune 模型
CXX=g++ TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3 /apdcephfs_cq2/share_1290939/branchwang/projects/E4S/Face_swap_with_two_imgs.py
# CXX=g++ TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3 /apdcephfs_cq2/share_1290939/branchwang/projects/E4S/our_swap_face_video_pipeline2.py
# CXX=g++ TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3 /apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video.py
# CXX=g++ python3 /apdcephfs_cq2/share_1290939/branchwang/projects/E4S/comp_images.py
# CXX=g++ TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3 /apdcephfs_cq2/share_1290939/branchwang/projects/E4S/write_video.py
# CXX=g++ TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3 /apdcephfs_cq2/share_1290939/branchwang/projects/E4S/crop_video.py

# 单卡 视频编辑
# CXX=g++ TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3 Face_swap_frontal.py
# CXX=g++ python3 /apdcephfs_cq2/share_1290939/branchwang/projects/E4S/Face_swap_with_two_imgs.py

# FF++ 数据集换脸
# CXX=g++ TORCH_HOME=/apdcephfs/share_1290939/zhianliu/pretrained_models/torch python3 FFPP_swap_face.py