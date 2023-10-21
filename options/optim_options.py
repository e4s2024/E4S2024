from argparse import ArgumentParser


class OptimOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, default="/apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/dummy",help='Path to experiment output directory')
		self.parser.add_argument('--num_seg_cls', type=int, default=12,help='Segmentation mask class number')
		self.parser.add_argument('--begin_idx', type=int, default=408)
		self.parser.add_argument('--remaining_layer_idx', type=int, default=13, help='剩余的几层不用mask')
  
        # ================= 模型设置 相关 =====================
		self.parser.add_argument('--out_size', type=int, default=1024,help='output image size')      
		self.parser.add_argument('--load_ema', default=False, type=bool, help='Whether to load the styleGAN EMA model')
		self.parser.add_argument('--n_styles', default=18, type=int, help='StyleGAN层数')
		self.parser.add_argument('--fsencoder_type', type=str, default="psp", help='FS Encode网络类型')    
		self.parser.add_argument('--checkpoint_path', default="/apdcephfs_cq2/share_1290939/branchwang/projects/pytorch-DDP-demo/iteration_300000.pt", type=str, help='Path to model checkpoint')
		self.parser.add_argument('--train_G', default=False, type=bool, help='Whether to train the styleGAN model')
  
        # ================= 数据集 相关 =====================
		self.parser.add_argument('--dataset_root', default='/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ', type=str, help='dataset root path')
		self.parser.add_argument('--ds_frac', default=1.0, type=float, help='dataset fraction')
		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=4, type=int, help='Number of test/inference dataloader workers')
		
		self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
		self.parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU(s) to use')

		self.parser.add_argument('--start_from_latent_avg', action='store_true',default=True, help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')
        # ================= 优化 相关 =========================
		self.parser.add_argument('--num_layers', default=18, type=int)
		self.parser.add_argument('--lr', default=1e-2, type=float)
		self.parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
		self.parser.add_argument('--W_steps', type=int, default=150, help='Number of W space optimization steps')
		self.parser.add_argument('--finetune_steps', type=int, default=100, help='Number of net finetune steps')
		self.parser.add_argument('--finetune_learning_rate', type=float, default=1e-3)
		self.parser.add_argument('--save_intermediate', action='store_true', help='Whether to store and save intermediate images during optimization')
		self.parser.add_argument('--verbose', action='store_true', help='Whether to verbose during optimization')  
		self.parser.add_argument('--save_interval', type=int, default=100, help='Latent checkpoint interval')
		self.parser.add_argument('--output_dir', type=str, default='/apdcephfs/share_1290939/zhianliu/running_results/our_editing/tmp', help='Optimizer output dir')
    
        # ================= Loss 相关 =====================
		self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--id_loss_multiscale', default=True, type=bool, help='Whether to apply multi scale in ID loss')  
		self.parser.add_argument('--face_parsing_lambda', default=0.1, type=float, help='Face parsing loss multiplier factor')		
		self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
		# self.parser.add_argument('--ir_se50_path', default='/apdcephfs_cq2/share_1290939/branchwang/pretrained_models/pixel2style2pixel/model_ir_se50.pth', type=str, help='Path to ir_se50 model weights')
		self.parser.add_argument('--ir_se50_path',
								 default='./pretrained/pixel2style2pixel/model_ir_se50.pth',
								 type=str, help='Path to ir_se50 model weights')
		self.parser.add_argument('--face_parsing_model_path', default='/apdcephfs_cq2/share_1290939/branchwang/pretrained_models/CelebA-Mask-HQ-faceParser/model.pth', type=str, help='Path to face parsing model weights')
	
	def parse(self):
		opts = self.parser.parse_args()
		return opts
