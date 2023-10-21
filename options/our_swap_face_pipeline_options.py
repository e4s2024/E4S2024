from argparse import ArgumentParser


class OurSwapFacePipelineOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, default="./tmp_exp", help='Path to experiment output directory')
		self.parser.add_argument('--num_seg_cls', type=int, default=12,help='Segmentation mask class number')
		self.parser.add_argument('--source_frame_name', type=str, default="28494", help='source frame number')
		self.parser.add_argument('--target_video_name', type=str, default="874",help='target video name')
        # ================= 模型设置 相关 =====================
		self.parser.add_argument('--out_size', type=int, default=1024, help='output image size') 
		self.parser.add_argument('--fsencoder_type', type=str, default="psp", help='FS Encode网络类型') 
		self.parser.add_argument('--remaining_layer_idx', type=int, default=13, help='剩余的几层不用mask')
		self.parser.add_argument('--outer_dilation', type=int, default=15, help='dilation 的宽度')
		self.parser.add_argument('--erode_radius', type=int, default=3, help='erode 的宽度')
    
        # ================= 数据集 相关 =====================
		self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--target_images_dir', default='/apdcephfs/share_1290939/zhianliu/py_projects/our_editing_swappingFace_video/01_2_face', type=str)
		self.parser.add_argument('--driven_images_dir', default='/apdcephfs/share_1290939/zhianliu/py_projects/our_editing_swappingFace_video/29698_to_01/driven', type=str)
		self.parser.add_argument('--UI_edit_masks_dir', default='/apdcephfs/share_1290939/zhianliu/py_projects/our_editing_swappingFace_video/29698_to_01/edit_mask', type=str)
		self.parser.add_argument('--swapped_style_vectors_dir', default='/apdcephfs/share_1290939/zhianliu/py_projects/our_editing_swappingFace_video/29698_to_01/FFHQ_model_video_swap_styleVec', type=str)

        # ================= 训练 相关 =====================
		self.parser.add_argument('--train_G', default=True, type=bool, help='Whether to train the model')
		self.parser.add_argument('--pti_learning_rate', default=1e-3, type=float, help='PTI learning rate')
		self.parser.add_argument('--stiching_learning_rate', default=1e-2, type=float, help='Stiching learning rate')
		self.parser.add_argument('--optim_name', default='adam', type=str, help='Which optimizer to use')  
		self.parser.add_argument('--max_pti_steps', default=0, type=int, help='PTI finetune steps')
		self.parser.add_argument('--max_stiching_steps', default=100, type=int, help='Stiching finetune steps')    
		self.parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU(s) to use')
  
        # ================= Loss 相关 =====================
		self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--id_loss_multiscale', default=True, type=bool, help='Whether to apply multi scale in ID loss')  
		self.parser.add_argument('--face_parsing_lambda', default=0.1, type=float, help='Face parsing loss multiplier factor')
		self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--recolor_lambda', default=5.0, type=float, help='Recolor reg loss multiplier factor')
  
        # ================== 预训练模型 相关 ==================
		self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')
        # 是否从styleGAN的均值开始学习
		self.parser.add_argument('--start_from_latent_avg', action='store_true',default=True, help='Whether to add average latent vector to generate codes from encoder.')
        # styleGAN输出图片大小
		self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
		self.parser.add_argument('--n_styles', default=18, type=int, help='StyleGAN层数')
  
        # ir_se50 预训练权重, for id_loss
		# self.parser.add_argument('--ir_se50_path', default='/apdcephfs/share_1290939/zhianliu/pretrained_models/pixel2style2pixel/model_ir_se50.pth', type=str, help='Path to ir_se50 model weights')
		self.parser.add_argument('--ir_se50_path',
								 default='./pretrained/pixel2style2pixel/model_ir_se50.pth',
								 type=str, help='Path to ir_se50 model weights')
		# self.parser.add_argument('--face_parsing_model_path', default='/apdcephfs/share_1290939/zhianliu/pretrained_models/CelebA-Mask-HQ-faceParser/model.pth', type=str, help='Path to face parsing model weights')
		self.parser.add_argument('--face_parsing_model_path',
								 default='./pretrained/faceseg/model.pth',
								 type=str, help='Path to face parsing model weights')
		# self.parser.add_argument('--checkpoint_path', default='/apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/v_15_hybrid_stage1_seg12_finetuneGD_8A100_pspHyperParas_remainLyrIdx13_flip_FFHQ_300KIters/checkpoints/iteration_300000_belowPyTorch1_6.pt', type=str, help='Path to model checkpoint')
		# self.parser.add_argument('--checkpoint_path', default='/apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/v_15_hybrid_stage1_seg12_finetuneGD_8A100_pspHyperParas_remainLyrIdx13_flip_FFHQ_300KIters/checkpoints/iteration_300000.pt', type=str, help='Path to model checkpoint')
		self.parser.add_argument('--checkpoint_path', default='./pretrained/E4S/iteration_300000.pt', type=str, help='Path to model checkpoint')
		self.parser.add_argument('--PTI_checkpoint_path', default='/apdcephfs/share_1290939/zhianliu/py_projects/pytorch-DDP-demo/work_dirs/v_18_video_swapping/musk_to_874/finetuned_G_lr1e3_iters150_erode.pth', type=str, help='Path to PTI finetuned model checkpoint')
		# self.parser.add_argument('--checkpoint_path', default='/apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/v_15_hybrid_stage1_seg12_finetuneGD_8A100_pspHyperParas_remainLyrIdx13_flip_200KIters/checkpoints/iteration_120000.pt', type=str, help='Path to model checkpoint')

		
	def parse(self):
		opts = self.parser.parse_args()
		return opts
