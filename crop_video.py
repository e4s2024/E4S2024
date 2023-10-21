import cv2
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from torch.nn import functional as F
import glob
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import gaussian

from utils.alignment import crop_faces, calc_alignment_coefficients


def save_image(image, output_folder, image_name, image_index, ext='jpg'):
    if ext == 'jpeg' or ext == 'jpg':
        image = image.convert('RGB')
    folder = os.path.join(output_folder, image_name)
    os.makedirs(folder, exist_ok=True)
    image.save(os.path.join(folder, '%04d.%s'%(image_index, ext)))


def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image

def save_frames_into_video(frames, video_path, fps=25):
    """
    将视频帧保存为文件
    
    args:
        frames [List of PIL.Image]: 视频帧
    """
    w, h = frames[0].width, frames[0].height

    print(len(frames))
    print((w, h))
    
    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 24, (w, h))
    for frm in frames:
        out.write(np.array(frm)[:, :, ::-1])
    
    # 完成工作后释放所有内容
    out.release()
    
def get_target_video(video_path, num_max_frames=50):
    
    # 读取视频参考了 https://cloud.tencent.com/developer/article/1687415
   
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    while cap.isOpened():
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret or len(frames) >= num_max_frames:
            print("Exiting ...")
            break
        frames.append(Image.fromarray(frame[:,:,::-1]))
    cap.release()
    
    assert len(frames) <= num_max_frames, "读取视频失败!"
    return frames, fps

def crop_video_follow_STIT(): # 用stich in time的方法提取视频中的人脸区域
    frames, fps = get_target_video("/apdcephfs/share_1290939/zhianliu/py_projects/our_editing_swappingFace_video/02_1.mp4", num_max_frames = 200)
    # files = sorted(glob.glob("/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video_res/result_hires/target_frames/*.jpg"))
    # files = sorted(glob.glob("/apdcephfs/share_1290939/zhianliu/datasets/video_swapping_demo/274/*.png"))[149:149+200]
    # files = sorted(glob.glob("/apdcephfs/share_1290939/zhianliu/datasets/video_swapping_demo/874/*.png"))[:200]
    # files = ["./halfFace.jpg"]
    # files = ["/apdcephfs/share_1290939/zhianliu/datasets/video_swapping_demo/celebrate/1.png",
    #          "/apdcephfs/share_1290939/zhianliu/datasets/video_swapping_demo/celebrate/2.png"]
    # files = [(os.path.basename(f).split('.')[0], f) for f in files]、
    files = [('_', f) for f in frames]
        
    image_size = 1024
    scale = 1.0
    center_sigma = 1.0
    xy_sigma = 3.0
    use_fa = False
    # output_folder = "/apdcephfs/share_1290939/zhianliu/datasets/video_swapping_demo/274/crop"
    # output_folder = "/apdcephfs/share_1290939/zhianliu/py_projects/tgtSkin_srcFaceShape_tgtMouth"
    output_folder = "/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video_res/swap_scarlett_to_target3"
    os.makedirs(output_folder, exist_ok=True)    
    
    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)

    '''
    os.makedirs(output_folder + '/crop', exist_ok=True) 
    os.makedirs(output_folder + '/target_frames', exist_ok=True) 
    for i in tqdm(range(len(crops)), total=len(crops)):
        crops[i].save(os.path.join(output_folder, "crop/%04d.png" % i))
        frames[i].save(os.path.join(output_folder, 'target_frames/%04d.png' % i))
    print('Aligning completed')
    '''
    # 贴回去
    
    edit_images = []
    for i in range(len(crops)):
        edit_images.append(Image.open(os.path.join(output_folder, "results/swap_face_%04d.png"%i)))
        
    inverse_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]

    res_frames = []
    for i, (coeffs, edit, orig_image) in tqdm(enumerate(zip(inverse_transforms, edit_images, orig_images)), total=len(orig_images)):
        
        pasted_image = paste_image(coeffs, edit, orig_image)  
        res_frames.append(pasted_image)
        save_image(pasted_image, output_folder, 'pasted', i)
    
    save_frames_into_video(res_frames, os.path.join(output_folder, 'res.mp4'), fps)
    
    


def crop_imgs(files): 
    
    files = [(os.path.basename(f).split('.')[0], f) for f in files]
        
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False
    
    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    for i in range(len(crops)):
        crops[i].save("./tmp/%d.png"%i)
    print('Aligning completed')
    

def anti_aliasing_test():  # 下采样抗锯齿测试

    src_img_dir = "/apdcephfs/share_1290939/zhianliu/datasets/CelebA-HQ/test/images"
    idx = [1989,61]
    img_names = ["%05d.jpg"%(i+28000) for i in idx]
    source_names = [os.path.join(src_img_dir, name) for name in img_names]
    
    for i in range(len(source_names)):
        # source_img = Image.open(source_names[i]).convert("RGB")  # 1024
        
        # imageio
        source_img = imageio.imread(source_names[i])  # 1024, [0, 1]
        
        source_img_blurred = gaussian(source_img, sigma=3, multichannel=True)
        source_img_blurred_256 = resize(source_img_blurred, (256, 256))
        source_img_blurred_256 = (source_img_blurred_256*255.0).astype(np.uint8)
        # Image.fromarray(source_img_blurred_256).save("%s_256_anti.png"%os.path.basename(source_names[i]).split('.')[0])
        
        source_img_256 = resize(source_img, (256, 256))
        source_img_256 = (source_img_256*255.0).astype(np.uint8)
        # Image.fromarray(source_img_256).save("%s_256.png"%os.path.basename(source_names[i]).split('.')[0])
        
        Image.fromarray(np.hstack((source_img_256, source_img_blurred_256))).save("%s.png"%os.path.basename(source_names[i]).split('.')[0])
        
                
if __name__=="__main__":
    crop_video_follow_STIT()

    # frame_path = sorted(glob.glob('/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video_res/swap_elon_to_353/pasted/' + '*.jpg'))
    # frames = []
    # for f in frame_path:
    #     frames.append(Image.open(f))

    # save_frames_into_video(frames, '/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video_res/swap_elon_to_353/res.mp4')
    
    # imgs = glob.glob("/apdcephfs/share_1290939/zhianliu/datasets/video_swapping_demo/celebrate/*")
    # crop_imgs(imgs)