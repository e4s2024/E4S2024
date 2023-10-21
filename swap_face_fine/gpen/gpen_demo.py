'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import glob
import time
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
# import __init_paths
from swap_face_fine.gpen.face_enhancement import FaceEnhancement

# ================== 以下是自己 DIY 的 ============================
class GPENInfer(object):
    def __init__(self):
        self.model = init_gpen_pretrained_model()

    def infer_image(self, img_lq: Image):
        img_lq = np.array(img_lq)
        img_hq = GPEN_demo(img_lq, self.model, aligned=False)
        img_hq = Image.fromarray(img_hq)
        return img_hq


def init_gpen_pretrained_model():
    """
    实例化 预训练的 GPEN 模型
    
    """
    default_model_params = {
        "base_dir": "./pretrained/GPEN/",  # 预训练参数所在根目录，注意，这个目录下面会有 ./weights 文件夹用于存放预训练模型参数文件
        "in_size": 512,
        "model": "GPEN-BFR-512", 
        "use_sr": True,
        "sr_model": "realesrnet",
        "sr_scale": 4,
        "channel_multiplier": 2,
        "narrow": 1,
    }
    model = FaceEnhancement(base_dir=default_model_params["base_dir"],
                            in_size=default_model_params["in_size"], 
                            model=default_model_params["model"], 
                            use_sr=default_model_params["use_sr"], 
                            sr_model=default_model_params["sr_model"],
                            sr_scale=default_model_params["sr_scale"],
                            channel_multiplier=default_model_params["channel_multiplier"],
                            narrow=default_model_params["narrow"], 
                            key=None, device='cuda')
    
    print("Load GPEN pre-traiend model success!")
    
    return model


def GPEN_demo(img, model, aligned=False, alpha=1.0):
    """ 
    驱动 source image
    
    args:
        img (np.array): [H,W,3] 256*256 大小, [0,255]范围, cv2 格式的 BGR顺序图片
        model (): FaceEnhancement 对象
        aligned (bool): 输入的人脸是否已经对齐
    return:
    
        img_out (np.array): [H,W,3] 256*256 大小, [0,255]范围, cv2 格式的 BGR顺序图片
    """
    with torch.no_grad():
        # img_out是整张图片的输入，orig_faces和enhanced_faces分别是增强前后的脸部区域
        img_out, orig_faces, enhanced_faces = model.process(img, aligned=aligned)
        img = cv2.resize(img, img_out.shape[:2])
        img_out = (img_out * alpha + img * (1 - alpha)).clip(0, 255).astype(img_out.dtype)
        
    return img_out    
            
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GPEN-BFR-512', help='GPEN model')
    parser.add_argument('--task', type=str, default='FaceEnhancement', help='task of GPEN model')
    parser.add_argument('--key', type=str, default=None, help='key of GPEN model')
    parser.add_argument('--in_size', type=int, default=512, help='in resolution of GPEN')
    parser.add_argument('--out_size', type=int, default=None, help='out resolution of GPEN')
    parser.add_argument('--channel_multiplier', type=int, default=2, help='channel multiplier of GPEN')
    parser.add_argument('--narrow', type=float, default=1, help='channel narrow scale')
    parser.add_argument('--use_sr', action='store_true', help='use sr or not')
    parser.add_argument('--use_cuda', action='store_true', help='use cuda or not')
    parser.add_argument('--save_face', action='store_true', help='save face or not')
    parser.add_argument('--aligned', action='store_true', help='input are aligned faces or not')
    parser.add_argument('--sr_model', type=str, default='realesrnet', help='SR model')
    parser.add_argument('--sr_scale', type=int, default=2, help='SR scale')
    parser.add_argument('--indir', type=str, default='examples/imgs', help='input folder')
    parser.add_argument('--outdir', type=str, default='results/outs-BFR', help='output folder')
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    processer = FaceEnhancement(in_size=args.in_size, model=args.model, use_sr=args.use_sr, sr_model=args.sr_model, sr_scale=args.sr_scale, channel_multiplier=args.channel_multiplier, narrow=args.narrow, key=args.key, device='cuda' if args.use_cuda else 'cpu')
    
    files = sorted(glob.glob(os.path.join(args.indir, '*.*g')))
    for n, file in enumerate(files[:]):
        filename = os.path.basename(file)
        
        img = cv2.imread(file, cv2.IMREAD_COLOR) # BGR
        if not isinstance(img, np.ndarray): print(filename, 'error'); continue
        #img = cv2.resize(img, (0,0), fx=2, fy=2) # optional

        img_out, orig_faces, enhanced_faces = processer.process(img, aligned=args.aligned)
        
        img = cv2.resize(img, img_out.shape[:2][::-1])
        cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1])+'_COMP.jpg'), np.hstack((img, img_out)))
        cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1])+'_GPEN.jpg'), img_out)
        
        if args.save_face:
            for m, (ef, of) in enumerate(zip(enhanced_faces, orig_faces)):
                of = cv2.resize(of, ef.shape[:2])
                cv2.imwrite(os.path.join(args.outdir, '.'.join(filename.split('.')[:-1])+'_face%02d'%m+'.jpg'), np.hstack((of, ef)))
        
        if n%10==0: print(n, filename)
