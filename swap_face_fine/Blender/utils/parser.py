import argparse


def get_base_parser():
    parser = argparse.ArgumentParser(description='PyTorch Face Blending')

    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--data_path', type=str, default='/apdcephfs_cq2/share_1290939/byschen/dataspace/vgg_face2_imgs')
    parser.add_argument('--eval_path', type=str, default='/apdcephfs_cq2/share_1290939/byschen/dataspace/swapping_pairs')
    parser.add_argument('--eval_img_ratio', default=0.1, type=float)

    parser.add_argument('--norm_G', type=str, default='spectralspadeinstance3x3',
                        help='instance normalization or batch normalization')
    parser.add_argument('--eqlr_sn', action='store_true', help='if true, use equlr, else use sn')
    parser.add_argument('--norm_E', type=str, default='spectralinstance',
                        help='instance normalization or batch normalization')
    parser.add_argument('--adaptor_kernel', type=int, default=3, help='kernel size in domain adaptor')
    parser.add_argument('--warp_stride', type=int, default=4, help='corr matrix 256 / warp_stride')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--adaptor_nonlocal', action='store_true', help='if true, use nonlocal block in domain adaptor')
    parser.add_argument('--adaptor_se', action='store_true', help='if true, use se layer in domain adaptor')
    parser.add_argument('--adaptor_res_deeper', action='store_true', help='if true, use 6 res block in domain adaptor')
    parser.add_argument('--dilation_conv', action='store_true',
                        help='if true, use dilation conv in domain adaptor when adaptor_res_deeper is True')
    parser.add_argument('--PONO', action='store_true', help='use positional normalization ')
    parser.add_argument('--PONO_C', action='store_true', help='use C normalization in corr module')
    parser.add_argument('--CBN_intype', type=str, default='warp_mask',
                        help='type of CBN input for framework, warp/mask/warp_mask')
    return parser


def add_base_train(parser):
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=-1, type=int)
    parser.add_argument('--optim', default='sgd', type=str)

    parser.add_argument('--iter_num', default=100, type=int)

    parser.add_argument('--lr_interval', default=20, type=int)
    parser.add_argument('--lr_decay', default=0.5, type=float)

    parser.add_argument('--report_interval', default=100, type=int)

    parser.add_argument('--eval_interval', default=10000, type=int)

    parser.add_argument('--vis_interval', default=500, type=int)

    parser.add_argument('--save_interval', default=5000, type=int)
    parser.add_argument('--resume', default='', type=str)
    return parser
