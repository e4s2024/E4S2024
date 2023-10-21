# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse


def str2bool(v):
    # print(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_opt():
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters

    parser.add_argument('--erode', type=int, default=19, help='')
    parser.add_argument('--lr_decay_interval', type=int, default=1, help='')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='')

    parser.add_argument('--train_sample_interval', type=int, default=1, help='')
    parser.add_argument('--eval_sample_interval', type=int, default=1, help='')
    parser.add_argument('--test_sample_interval', type=int, default=1, help='')

    parser.add_argument('--eval_interval', type=int, default=2, help ='saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_mode_interval', type=int, default=1, help='')

    parser.add_argument('--train_batch_size', type=int, default=1, help='size of the batches')

    parser.add_argument('--train_data', type=str, default='./data/face.txt', help='images baseroot')
    parser.add_argument('--train_mask', type=str, default='./data/mask_20.txt', help='images baseroot')
    parser.add_argument('--eval_data', type=str, default='./data/face.txt', help='images baseroot')
    parser.add_argument('--eval_mask', type=str, default='./data/mask_20.txt', help='images baseroot')
    parser.add_argument('--test_data', type=str, default='./data/dunhuang.txt', help='images baseroot')
    parser.add_argument('--test_mask', type=str, default='./data/dunhuang_mask.txt', help='images baseroot')

    parser.add_argument('--save_model', type=str, default='./result/model/', help='saving path that is a folder')
    parser.add_argument('--train_sample', type=str, default='./result/train_sample',
                        help='saving path that is a folder')
    parser.add_argument('--eval_sample', type=str, default='./result/eval_sample', help='saving path that is a folder')
    parser.add_argument('--test_sample', type=str, default='./result/test_sample', help='saving path that is a folder')

    parser.add_argument('--load_name', type=str, default='./result/model/251000_KPN_bs_8_opt.pth', help='loach')

    parser.add_argument('--dataset', type=int, default=1, help='1 dunhuang, 0 other')
    parser.add_argument('--mask_threshold', type=int, default=100, help='1 dunhuang, 0 other')

    parser.add_argument('--test_batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--crop', type=str2bool, default=False, help='whether to crop input images')

    # GPU parameters

    parser.add_argument('--multi_gpu', type=str2bool, default=False, help='True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=True, help='True for unchanged input data type')
    # Training parameters

    parser.add_argument('--lr_g', type=float, default=0.0002, help='Adam: learning rate for G / D')
    parser.add_argument('--b1', type=float, default=0.5, help='Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for optimizer')
    parser.add_argument('--lr_decrease_epoch', type=int, default=20,
                        help='lr decrease at certain epoch and its multiple')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of cpu threads to use during batch generation')
    # Initialization parameters
    parser.add_argument('--color', type=str2bool, default=True, help='input type')
    parser.add_argument('--burst_length', type=int, default=1, help='number of photos used in burst setting')
    parser.add_argument('--blind_est', type=str2bool, default=True, help='variance map')
    parser.add_argument('--kernel_size', type=str2bool, default=[3], help='kernel size')
    parser.add_argument('--sep_conv', type=str2bool, default=False, help='simple output type')
    parser.add_argument('--channel_att', type=str2bool, default=False, help='channel wise attention')
    parser.add_argument('--spatial_att', type=str2bool, default=False, help='spatial wise attention')
    parser.add_argument('--upMode', type=str, default='bilinear', help='upMode')
    parser.add_argument('--core_bias', type=str2bool, default=False, help='core_bias')
    parser.add_argument('--init_type', type=str, default='xavier', help='initialization type of generator')
    parser.add_argument('--init_gain', type=float, default=0.02, help='initialization gain of generator')

    # Dataset parameters
    parser.add_argument('--rainaug', type=str2bool, default=False, help='true for using rainaug')
    parser.add_argument('--crop_size', type=int, default=256, help='single patch size')
    parser.add_argument('--geometry_aug', type=str2bool, default=False, help='geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type=str2bool, default=True, help='geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type=float, default=1, help='min scaling factor')
    parser.add_argument('--scale_max', type=float, default=1, help='max scaling factor')
    parser.add_argument('--mu', type=int, default=0, help='Gaussian noise mean')
    parser.add_argument('--sigma', type=int, default=30, help='Gaussian noise variance: 30 | 50 | 70')
    opt = parser.parse_args()
    print(opt)
    return opt

