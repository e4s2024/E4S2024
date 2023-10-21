import os
import copy
import uuid
import glob
import shutil
from typing import Union, List
from pathlib import Path
from multiprocessing.pool import Pool

import torch
from torchvision import transforms
import torch.nn.functional as F

import cv2
import numpy as np
from PIL import Image
import tqdm

from utils.alignment import crop_faces, calc_alignment_coefficients
from utils.morphology import dilation, erosion


""" From MegaFS: https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels/tree/main/inference """
class SoftErosion(torch.nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


def read_video_as_frames(data_path: str,
                         out_frames_folder: str = None,
                         frame_period: int = 1
                         ) -> (List[np.ndarray], List[str]):
    if out_frames_folder is not None:
        os.makedirs(out_frames_folder, exist_ok=True)
    reader = cv2.VideoCapture(data_path)
    frame_num = 0
    ret_frames = []
    ret_paths = []
    while reader.isOpened():
        success, image = reader.read()  # BGR
        if not success:  # finished
            break
        if frame_num % frame_period != 0:  # skip
            continue
        if out_frames_folder is not None:
            save_path = os.path.join(out_frames_folder, 'frame_%05d.png' % frame_num)
            cv2.imwrite(save_path, image)
            ret_paths.append(save_path)
        ret_frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # RGB
        frame_num += 1
    reader.release()
    return ret_frames, ret_paths


def save_frames_as_video(frames: Union[list, str],
                         video_save_dir: str,
                         video_save_fn: str = "output.mp4",
                         frame_template: str = "frame_%05d.png",
                         fps: int = 25,
                         audio_from: str = None,
                         delete_tmp_frames: bool = False,
                         ):
    if isinstance(frames, str):
        frames_dir = frames
    elif isinstance(frames, list):
        frames_dir = os.path.join(video_save_dir, "tmp_frames")
        for idx, frame in enumerate(frames):
            frame.save(os.path.join(frames_dir, "frame_%05d.png" % idx))
    else:
        raise TypeError("Unsupported frames type.")

    os.makedirs(video_save_dir, exist_ok=True)
    video_save_path = os.path.join(video_save_dir, video_save_fn)

    if audio_from is not None:
        print("use audio")
        os.system(
            f"ffmpeg  -y -r {fps} -i {frames_dir}/{frame_template} -i {audio_from}"
            f" -map 0:v:0 -map 1:a:0? -c:a copy -c:v libx264 -r {fps} -crf 10 -pix_fmt yuv420p {video_save_path}"
        )
    else:
        print("no audio")
        os.system(
            f"ffmpeg  -y -r {fps} -i {frames_dir}/{frame_template} "
            f"-c:v libx264 -r {fps} -crf 10 -pix_fmt yuv420p {video_save_path}"
        )

    if delete_tmp_frames:
        shutil.rmtree(frames_dir)
        for match in glob.glob(os.path.join(frames_dir, "*.png")):
            os.remove(match)

    print(f"video saved to: {video_save_path}")
    return video_save_path


def paste_frames_to_frames(frames_up: str,
                           frames_down: str,
                           video_save_dir: str,
                           video_save_fn: str = "stable_output.mp4",
                           ):
    up_fns = os.listdir(frames_up)
    down_fns = os.listdir(frames_down)
    up_paths = [os.path.join(frames_up, fn) for fn in up_fns]
    down_paths = [os.path.join(frames_down, fn) for fn in down_fns]
    up_files = [(os.path.basename(f).split('.')[0], f) for f in up_paths]
    down_files = [(os.path.basename(f).split('.')[0], f) for f in down_paths]

    source_crops, source_orig_images, source_quads, source_inv_transforms = crop_and_align_face(
        up_files, image_size=1024, scale=1.0, center_sigma=0, xy_sigma=0, use_fa=False
    )
    S = [crop.convert("RGB") for crop in source_crops]
    # target_crops, target_orig_images, target_quads, target_inv_transforms = crop_and_align_face(
    #     down_files, image_size=1024, scale=1.0, center_sigma=1.0, xy_sigma=3.0, use_fa=False
    # )
    target_orig_images = [Image.open(down_path) for down_path in down_paths]
    # for i in range(len(down_files)):
    #     quad = source_quads[i]
    #
    # T = [crop.convert("RGB") for crop in target_crops]

    # paste back
    os.makedirs(video_save_dir, exist_ok=True)
    video_save_frames_dir = os.path.join(video_save_dir, "stable_frames")
    frames_fn_template = "%05d.png"
    os.makedirs(video_save_frames_dir, exist_ok=True)
    for i in range(len(S)):
        up = S[i].convert('RGBA')
        down = target_orig_images[i].convert('RGBA')
        up.putalpha(255)

        projected = up.transform(down.size, Image.PERSPECTIVE,
                                 source_inv_transforms[i],
                                 Image.BILINEAR)
        down.alpha_composite(projected)
        down.save(os.path.join(video_save_frames_dir, frames_fn_template % i))

    # save to video
    save_frames_as_video(
        video_save_frames_dir,
        video_save_dir,
        video_save_fn,
        frame_template=frames_fn_template
    )


def crop_frames(frames_folder: str,
                crops_folder: str = None,
                ):
    if crops_folder is None:
        crops_folder = f"{frames_folder}_crop"
    os.makedirs(crops_folder, exist_ok=True)
    fns = os.listdir(frames_folder)
    for i in tqdm.tqdm(range(len(fns))):
        frame = Image.open(os.path.join(frames_folder, fns[i]))
        w, h = frame.size
        frame = frame.crop((w - h, 0, w, h))
        frame.save(os.path.join(crops_folder, fns[i]))


def crop_and_align_face(target_files, image_size=1024, scale=1.0, center_sigma=1.0, xy_sigma=3.0, use_fa=False):
    print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, target_files, scale, center_sigma=center_sigma,
                                           xy_sigma=xy_sigma, use_fa=use_fa)

    # crop 的逆变换，用于后期贴回到原始视频上去
    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]

    return crops, orig_images, quads, inv_transforms


def logical_or_reduce(*tensors):
    return torch.stack(tensors, dim=0).any(dim=0)


def create_masks(mask, operation='dilation', radius=0):
    temp = copy.deepcopy(mask)
    if operation == 'dilation':
        full_mask = dilation(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = full_mask - temp
    elif operation == 'erosion':
        full_mask = erosion(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        border_mask = temp - full_mask
    # 'expansion' means to obtain a boundary that expands to both sides
    elif operation == 'expansion':
        full_mask = dilation(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device), engine='convolution')
        erosion_mask = erosion(temp, torch.ones(2 * radius + 1, 2 * radius + 1, device=mask.device),
                               engine='convolution')
        border_mask = full_mask - erosion_mask

    border_mask = border_mask.clip(0, 1)
    content_mask = mask

    return content_mask, border_mask, full_mask


def get_facial_mask_from_seg19(seg_map_long: torch.LongTensor,
                               target_size: tuple = None,
                               edge_softer: SoftErosion = None,
                               is_seg19: bool = False,
                               ):
    """ segmentation format:
        0 - background
        1 - lip
        2 - eyebrow
        3 - eyes
        4 - hair
        5 - nose
        6 - skin
        7 - ear
        8 - neck
        9 - tooth
        10 -
        11 - earring
    """
    if is_seg19:
        seg_map_long = torch.LongTensor(seg19_to_seg12(seg_map_long.cpu().numpy()))
    facial_indices = (1, 2, 3, 5, 6, 8, 9)
    mask = torch.zeros_like(seg_map_long, dtype=torch.long)
    for index in facial_indices:
        mask = mask + ((seg_map_long == index).long())  # either in {0,1}
    mask = mask.float()
    if target_size is not None:
        mask = F.interpolate(mask, size=target_size, mode="bilinear", align_corners=True)
    if edge_softer is not None:
        mask, _ = edge_softer(mask)
    return mask.cpu().numpy()


def seg19_to_seg12(mask):
    """将 ffhq 的mask (用face-parsing.PyTorch模型提取到的) 转为 faceParser格式（聚合某些 facial component）

    Args:
        mask (PIL image): with shape [H,W]
    Return:
        返回转化后的mask，同样是shape [H,W]，但是类别数更少
    """

    # assert len(mask.size) == 2, "mask 必须是[H,W]格式的数据"

    converted_mask = np.zeros_like(mask)

    backgorund = np.equal(mask, 0)
    converted_mask[backgorund] = 0

    lip = np.logical_or(np.equal(mask, 12), np.equal(mask, 13))  # 嘴唇
    converted_mask[lip] = 1

    eyebrows = np.logical_or(np.equal(mask, 2),
                             np.equal(mask, 3))
    converted_mask[eyebrows] = 2

    eyes = np.logical_or(np.equal(mask, 4), np.equal(mask, 5))
    converted_mask[eyes] = 3

    hair = np.equal(mask, 17)
    converted_mask[hair] = 4

    nose = np.equal(mask, 10)
    converted_mask[nose] = 5

    skin = np.equal(mask, 1)
    converted_mask[skin] = 6

    ears = np.logical_or(np.equal(mask, 7), np.equal(mask, 8))
    converted_mask[ears] = 7

    belowface = np.equal(mask, 14)
    converted_mask[belowface] = 8

    mouth = np.equal(mask, 11)  # 牙齿
    converted_mask[mouth] = 9

    eye_glass = np.equal(mask, 6)  # 眼镜
    converted_mask[eye_glass] = 10

    ear_rings = np.equal(mask, 9)  # 耳环
    converted_mask[ear_rings] = 11

    return converted_mask


def get_edge(img: Image, threshold: int = 128) -> Image:
    img = np.array(img).astype(np.uint8)
    img_x = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
    img_y = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))
    edge = cv2.addWeighted(img_x, 1, img_y, 1, 0)
    edge = cv2.cvtColor(edge, cv2.COLOR_RGB2GRAY)
    pos_big = np.where(edge >= threshold)
    pos_small = np.where(edge < threshold)
    # edge = cv2.GaussianBlur(edge, ksize=(3, 3), sigmaX=5)
    # edge[pos_big] = (edge[pos_big] * 1.05).clip(0, 255)
    # edge[pos_small] = (edge[pos_small] / 1.).clip(0, 255)
    # edge = cv2.GaussianBlur(edge, ksize=(5, 5), sigmaX=11)
    return Image.fromarray(edge.astype(np.uint8))


def blending_two_images_with_mask(bottom: Image, up: Image,
                                  up_ratio: float = 1.0,
                                  up_mask: np.ndarray = None
                                  ) -> Image:
    h, w = bottom.size
    if up_mask is None:
        up_mask = np.ones((h, w, 1), dtype=float)
    else:
        up_mask = up_mask.squeeze()[:, :, None]
    up_mask[np.isnan(up_mask)] = 0.  # input may contain NAN
    assert 0.0 <= up_ratio <= 1.0, "Blending Ratio should be in [0.0, 1.0]!"
    up_mask *= up_ratio
    i_a = np.array(bottom)
    i_b = np.array(up)
    i_a = i_a * (1 - up_mask) + i_b * up_mask
    ret_image = Image.fromarray(i_a.astype(np.uint8).clip(0, 255))
    return ret_image


if __name__ == "__main__":
    # stit_folder = "/home/yuange/datasets/STIT/datasets"
    # names = os.listdir(stit_folder)
    # for name in names:
    #     if not os.path.isdir(os.path.join(stit_folder, name)):
    #         continue
    #     save_frames_as_video(
    #         os.path.join(stit_folder, name),
    #         stit_folder,
    #         f"{name}.mp4",
    #         frame_template="%4d.jpeg",
    #         delete_tmp_frames=False,
    #     )

    # save_frames_as_video(
    #     "/home/yuange/program/PyCharmRemote/All-In-One-Deflicker/results/output/final/output",
    #     "/home/yuange/program/PyCharmRemote/All-In-One-Deflicker/results/output/final/",
    #     "ours_cat.mp4",
    #     frame_template="%05d.png",
    # )

    # paste_frames_to_frames(
    #     frames_up="/home/yuange/program/PyCharmRemote/All-In-One-Deflicker/results/jim_swap/final/output",
    #     frames_down="/home/yuange/program/PyCharmRemote/All-In-One-Deflicker/inputs/jim_swap",
    #     video_save_dir="/home/yuange/program/PyCharmRemote/All-In-One-Deflicker/results/jim_swap",
    # )

    to_be_cropped = [
        "/home/yuange/Documents/E4S_v2/sota_video_results/e4s_v2/michael_swap/output",
        "/home/yuange/Documents/E4S_v2/sota_video_results/e4s_v2/jim_swap/output",
        "/home/yuange/Documents/E4S_v2/sota_video_results/infoswap/target/00000",
        "/home/yuange/Documents/E4S_v2/sota_video_results/infoswap/target/00001",
        "/home/yuange/Documents/E4S_v2/sota_video_results/infoswap/result/00000",
        "/home/yuange/Documents/E4S_v2/sota_video_results/infoswap/result/00001",
        "/home/yuange/Documents/E4S_v2/sota_video_results/hires/result/00000",
        "/home/yuange/Documents/E4S_v2/sota_video_results/hires/result/00001",
        "/home/yuange/Documents/E4S_v2/sota_video_results/faceshifter/result/00000",
        "/home/yuange/Documents/E4S_v2/sota_video_results/faceshifter/result/00001",
        "/home/yuange/Documents/E4S_v2/sota_video_results/simswap/result/00000",
        "/home/yuange/Documents/E4S_v2/sota_video_results/simswap/result/00001",
    ]
    for task in to_be_cropped:
        crop_frames(
            task
        )

    # img_down = Image.open("recolor_input_0000.png")
    # img_up = Image.open("enhance_0000.png")
    #
    # from swap_face_fine.realesr.image_infer import RealESRBatchInfer
    # super_res = RealESRBatchInfer()
    # img_up = super_res.infer_image(img_up)
    # img_up.save("super_res.png")
    #
    # blending_mask = np.ones(img_down.size, dtype=np.float32)
    # edge = get_edge(img_down)
    # edge = np.array(edge).astype(np.float32) / 255.
    # blending_mask = (blending_mask - edge).clip(0., 1.)
    # # blending_mask = cv2.dilate(blending_mask, np.ones((2, 2), np.uint8), iterations=1)
    # Image.fromarray((blending_mask.squeeze() * 255.).astype(np.uint8)).save(
    #     "blend_mask.png"
    # )
    # blended = blending_two_images_with_mask(
    #     img_down, img_up, up_ratio=0.95, up_mask=blending_mask.copy()
    # )
    # blended.save("blend_result.png")
