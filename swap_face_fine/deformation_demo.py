#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jian-Wei ZHANG
@email: zjw.cs@zju.edu.cn
@date: 2017/8/8
@update: 2020/9/25
@update: 2021/7/14: Simplify usage
@update: 2021/12/24: Fix bugs and add an example of random control points (see `demo2()`)
"""

import time

import numpy as np
import math
import matplotlib.pyplot as plt

from .landmark import get_landmark
from .landmark_smooth import kalman_filter

try:
    import torch    # Install PyTorch first: https://pytorch.org/get-started/locally/
    from .img_utils_pytorch import (
        mls_affine_deformation as mls_affine_deformation_pt,
        mls_similarity_deformation as mls_similarity_deformation_pt,
        mls_rigid_deformation as mls_rigid_deformation_pt,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
except ImportError as e:
    print(e)

from .img_utils import (
    mls_affine_deformation, 
    mls_similarity_deformation, 
    mls_rigid_deformation
)

from PIL import Image


def demo():
    """
    p = np.array([
        [155, 30], [155, 125], [155, 225],
        [235, 100], [235, 160], [295, 85], [293, 180]
    ])
    q = np.array([
        [211, 42], [155, 125], [100, 235],
        [235, 80], [235, 140], [295, 85], [295, 180]
    ])
    
    image = np.array(Image.open("images/toy.jpg"))
    """

    source_path = '/apdcephfs_cq2/share_1290939/branchwang/data/e4s_vis/f-elegant-1_to_f2_faceVid2Vid/S_cropped.png'
    target_path = '/apdcephfs_cq2/share_1290939/branchwang/data/e4s_vis/f-elegant-1_to_f2_faceVid2Vid/T_cropped.png'
    image = Image.open(target_path)
    image_ref = Image.open(source_path)

    lm_face = get_landmark(image)
    lm_ref = get_landmark(image_ref)
    p = lm_face[:17][:, ::-1]
    q = lm_ref[:17][:, ::-1]

    image = np.array(image)

    height, width, _ = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)
    
    affine = mls_affine_deformation(vy, vx, p, q, alpha=1)
    aug1 = np.ones_like(image)
    aug1[vx, vy] = image[tuple(affine)]

    similar = mls_similarity_deformation(vy, vx, p, q, alpha=1)
    aug2 = np.ones_like(image)
    aug2[vx, vy] = image[tuple(similar)]

    rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
    aug3 = np.ones_like(image)
    aug3[vx, vy] = image[tuple(rigid)]

    fig, ax = plt.subplots(1, 5, figsize=(12, 5))
    ax[0].imshow(image)
    ax[0].scatter(lm_face[: 17][:, 0], lm_face[: 17][:, 1], c='b', s=2)
    ax[0].scatter(lm_ref[: 17][:, 0], lm_ref[: 17][:, 1], c='r', s=2)
    ax[0].set_title("Original Image")  

    ax[1].imshow(aug1)
    ax[1].scatter(lm_face[: 17][:, 0], lm_face[: 17][:, 1], c='b', s=2)
    ax[1].scatter(lm_ref[: 17][:, 0], lm_ref[: 17][:, 1], c='r', s=2)
    ax[1].set_title("Affine Deformation")

    ax[2].imshow(aug2)
    ax[2].scatter(lm_face[: 17][:, 0], lm_face[: 17][:, 1], c='b', s=2)
    ax[2].scatter(lm_ref[: 17][:, 0], lm_ref[: 17][:, 1], c='r', s=2)
    ax[2].set_title("Similarity Deformation")

    ax[3].imshow(aug3)
    ax[3].scatter(lm_face[: 17][:, 0], lm_face[: 17][:, 1], c='b', s=2)
    ax[3].scatter(lm_ref[: 17][:, 0], lm_ref[: 17][:, 1], c='r', s=2)
    ax[3].set_title("Rigid Deformation")

    ax[4].imshow(np.array(image_ref))
    ax[4].scatter(lm_ref[: 17][:, 0], lm_ref[: 17][:, 1], c='r', s=2)
    ax[4].set_title("target")

    for x in ax.flat:
        x.axis("off")

    plt.tight_layout(w_pad=0.1)
    plt.show()
    plt.savefig('images/face_results_numpy.png')
    Image.fromarray(aug3).save('images/face_rigid_deformation_numpy.png')


def demo_torch():
    """
    p = torch.from_numpy(np.array([
        [155, 30], [155, 125], [155, 225],
        [235, 100], [235, 160], [295, 85], [293, 180]
    ])).to(device)
    q = torch.from_numpy(np.array([
        [211, 42], [155, 125], [100, 235],
        [235, 80], [235, 140], [295, 85], [295, 180]
    ])).to(device)
    
    image = torch.from_numpy(np.array(Image.open("images/toy.jpg"))).to(device)
    """

    source_path = '/apdcephfs_cq2/share_1290939/branchwang/data/e4s_vis/f-elegant-1_to_f2_faceVid2Vid/S_cropped.png'
    target_path = '/apdcephfs_cq2/share_1290939/branchwang/data/e4s_vis/f-elegant-1_to_f2_faceVid2Vid/T_cropped.png'
    image = Image.open(target_path)
    image_ref = Image.open(source_path)

    lm_face = get_landmark(image)
    lm_ref = get_landmark(image_ref)
    p = lm_face[:, ::-1]
    q = lm_ref[:, ::-1]
    p = torch.from_numpy(p.copy()).to(device)
    q = torch.from_numpy(q.copy()).to(device)

    image = torch.from_numpy(np.array(image)).to(device)

    
    height, width, _ = image.shape
    gridX = torch.arange(width, dtype=torch.int16).to(device)
    gridY = torch.arange(height, dtype=torch.int16).to(device)
    vy, vx = torch.meshgrid(gridX, gridY)
    # !!! Pay attention !!!: the shape of returned tensors are different between numpy.meshgrid and torch.meshgrid
    vy, vx = vy.transpose(0, 1), vx.transpose(0, 1)
    
    affine = mls_affine_deformation_pt(vy, vx, p, q, alpha=1)
    aug1 = torch.ones_like(image).to(device)
    aug1[vx.long(), vy.long()] = image[tuple(affine)]

    similar = mls_similarity_deformation_pt(vy, vx, p, q, alpha=1)
    aug2 = torch.ones_like(image).to(device)
    aug2[vx.long(), vy.long()] = image[tuple(similar)]

    rigid = mls_rigid_deformation_pt(vy, vx, p, q, alpha=1)
    aug3 = torch.ones_like(image).to(device)
    aug3[vx.long(), vy.long()] = image[tuple(rigid)]

    mask = torch.ones_like(aug3, device=aug3.device)
    margin = 10
    mask[:margin, :, :] = 0
    mask[height-margin:, :, :] = 0
    mask[:, :margin, :] = 0
    mask[:, width-margin:, :] = 0

    """
    aug3_ori = aug3.clone()
    aug3 = aug3 * mask

    aug3 = aug3.permute(2, 0, 1).unsqueeze(0)
    radius = 7
    aug3 = dilation(aug3.float(), torch.ones(2 * radius + 1, 2 * radius + 1, device=aug3.device), engine='convolution')
    aug3 = aug3[0].permute(1, 2, 0).int()
    aug3 = aug3 * (1 - mask) + aug3_ori * mask
    """
    aug1 = aug1 * mask + image * (1 - mask)
    aug2 = aug2 * mask + image * (1 - mask)
    aug3 = aug3 * mask + image * (1 - mask)

    fig, ax = plt.subplots(1, 5, figsize=(12, 5))
    ax[0].imshow(image.cpu().numpy())
    ax[0].scatter(lm_face[: 17][:, 0], lm_face[: 17][:, 1], c='b', s=2)
    ax[0].scatter(lm_ref[: 17][:, 0], lm_ref[: 17][:, 1], c='r', s=2)
    ax[0].set_title("Original Image")  

    ax[1].imshow(aug1.cpu().numpy())
    ax[1].scatter(lm_face[: 17][:, 0], lm_face[: 17][:, 1], c='b', s=2)
    ax[1].scatter(lm_ref[: 17][:, 0], lm_ref[: 17][:, 1], c='r', s=2)
    ax[1].set_title("Affine Deformation")

    ax[2].imshow(aug2.cpu().numpy())
    ax[2].scatter(lm_face[: 17][:, 0], lm_face[: 17][:, 1], c='b', s=2)
    ax[2].scatter(lm_ref[: 17][:, 0], lm_ref[: 17][:, 1], c='r', s=2)
    ax[2].set_title("Similarity Deformation")

    ax[3].imshow(aug3.cpu().numpy())
    ax[3].scatter(lm_face[: 17][:, 0], lm_face[: 17][:, 1], c='b', s=2)
    ax[3].scatter(lm_ref[: 17][:, 0], lm_ref[: 17][:, 1], c='r', s=2)
    ax[3].set_title("Rigid Deformation")

    ax[4].imshow(np.array(image_ref))
    ax[4].scatter(lm_ref[: 17][:, 0], lm_ref[: 17][:, 1], c='r', s=2)
    ax[4].set_title("target")

    for x in ax.flat:
        x.axis("off")

    plt.tight_layout(w_pad=0.1)
    plt.show()
    plt.savefig('images/face_results.png')

    Image.fromarray(aug3.cpu().numpy()).save('images/face_rigid_deformation.png')


def demo2():
    """ Smiled Monalisa """
    np.random.seed(1234)
    
    image = np.array(Image.open("images/monalisa.jpg"))
    height, width, _ = image.shape
    
    # Define deformation grid
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    # ================ Control points group 1 (manually specified) ==================
    p1 = np.array([[0, 0], [517, 0], [0, 798], [517, 798],
        [140, 186], [135, 295], [181, 208], [181, 261], [203, 184], [202, 304], [225, 213], 
        [225, 243], [244, 211], [244, 253], [254, 195], [281, 232], [252, 285]
    ])
    q1 = np.array([[0, 0], [517, 0], [0, 798], [517, 798],
        [140, 186], [135, 295], [181, 208], [181, 261], [203, 184], [202, 304], [225, 213], 
        [225, 243], [238, 207], [237, 261], [253, 199], [281, 232], [249, 279]
    ])

    rigid1 = mls_rigid_deformation(vy, vx, p1, q1, alpha=1)
    aug1 = np.ones_like(image)
    aug1[vx, vy] = image[tuple(rigid1)]

    # ====================== Control points group 1 (random) =======================
    p2 = np.stack((
        np.random.randint(0, height, size=13), 
        np.random.randint(0, width, size=13),
    ), axis=1)
    q2 = p2 + np.random.randint(-20, 20, size=p2.shape)

    rigid2 = mls_rigid_deformation_pt(vy, vx, p2, q2, alpha=1)
    aug2 = np.ones_like(image)
    aug2[vx, vy] = image[tuple(rigid2)]

    fig, ax = plt.subplots(1, 3, figsize=(13, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(aug1)
    ax[1].set_title("Manually specified control points")
    ax[2].imshow(aug2)
    ax[2].set_title("Random control points")

    for x in ax.flat:
        x.axis("off")
    
    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()


def read_tif(frame):
    image_pil = Image.open("images/train-volume.tif")
    image_pil.seek(frame)
    image = np.array(image_pil)
    label_pil = Image.open("images/train-labels.tif")
    label_pil.seek(frame)
    label = np.array(label_pil)

    return image, label


def demo3():
    image, label = read_tif(1)
    image = np.pad(image, ((30, 30), (30, 30)), mode='symmetric')
    label = np.pad(label, ((30, 30), (30, 30)), mode='symmetric')

    height, width = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    def augment(p, q, mode='affine'):
        if mode.lower() == 'affine':
            transform = mls_affine_deformation(vy, vx, p, q, alpha=1)
        elif mode.lower() == 'similar':
            transform = mls_similarity_deformation(vy, vx, p, q, alpha=1)
        elif mode.lower() == 'rigid':
            transform = mls_rigid_deformation(vy, vx, p, q, alpha=1)
        else:
            raise ValueError

        aug_img = np.ones_like(image)
        aug_img[vx, vy] = image[tuple(transform)]
        aug_lab = np.ones_like(label)
        aug_lab[vx, vy] = label[tuple(transform)]

        return aug_img, aug_lab

    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title("Original Image")
    ax[1, 0].imshow(label, cmap='gray')
    ax[1, 0].set_title("Original Label")
    
    np.random.seed(1234)
    p = np.c_[np.random.randint(0, height, size=32), np.random.randint(0, width, size=32)]
    q = p + np.random.randint(-15, 15, size=p.shape)
    q[:, 0] = np.clip(q[:, 0], 0, height)
    q[:, 1] = np.clip(q[:, 1], 0, width)
    p = np.r_[p, np.array([[0, 0], [0, width - 1], [height - 1, 0], [height - 1, width - 1]])]  # fix corner points
    q = np.r_[q, np.array([[0, 0], [0, width - 1], [height - 1, 0], [height - 1, width - 1]])]  # fix corner points

    for i, mode in enumerate(['Affine', 'Similar', 'Rigid']):
        aug_img, aug_lab = augment(p, q, mode)
        ax[0, i + 1].imshow(aug_img, cmap='gray')
        ax[0, i + 1].set_title(f"{mode} Deformated Image")
        ax[1, i + 1].imshow(aug_lab, cmap='gray')
        ax[1, i + 1].set_title(f"{mode} Deformated Label")

    for x in ax.flat:
        x.axis('off')

    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()


def benchmark_numpy(image, p, q):
    height, width = image.shape[:2]

    # Define deformation grid
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
    aug = np.ones_like(image)
    aug[vx, vy] = image[tuple(rigid)]
    return aug


def benchmark_torch(image, p, q):
    height, width = image.shape[:2]
    device = image.device

    # Define deformation grid
    gridX = torch.arange(width, dtype=torch.int16).to(device)
    gridY = torch.arange(height, dtype=torch.int16).to(device)
    vy, vx = torch.meshgrid(gridX, gridY)

    rigid = mls_rigid_deformation_pt(vy, vx, p, q, alpha=1)
    aug = torch.ones_like(image).to(device)
    aug[vx.long(), vy.long()] = image[rigid[0], rigid[1]]
    return aug


def run_benckmark(i):
    sizes = [   # (height, width)
        (100, 100),
        (500, 500),
        (500, 500),
        (500, 500),
        (1000, 1000),
        (2000, 2000),
    ]
    num_pts = [16, 16, 32, 64, 64, 64]

    times = []
    for _ in range(3):
        image = np.random.randint(0, 256, sizes[i])
        height, width = image.shape[:2]
        p = np.stack((
            np.random.randint(0, height, size=num_pts[i]), 
            np.random.randint(0, width, size=num_pts[i]),
        ), axis=1)
        q = p + np.random.randint(-20, 20, size=p.shape)

        start = time.time()
        _ = benchmark_numpy(image, p, q)
        elapse = time.time() - start
        times.append(elapse)
    print("Time (numpy):", sum(times) / len(times))

    times = []
    for _ in range(3):
        image = torch.randint(0, 256, sizes[i]).to(device)
        height, width = image.shape[:2]
        p = torch.stack((
            torch.randint(0, height, size=(num_pts[i],)),
            torch.randint(0, width, size=(num_pts[i],)),
        ), dim=1).to(device)
        q = p + torch.randint(-20, 20, size=p.shape).to(device)

        start = time.time()
        _ = benchmark_torch(image, p, q)
        elapse = time.time() - start
        times.append(elapse)
    print("Time (torch):", sum(times) / len(times))


def interp(a, b):
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    points = []
    for t in alpha:
        points.append(a * t + b * (1 - t))

    return points


def get_fixed_points(lm, scale=1.):
    left_to_right = lm[16] - lm[0]
    top_to_down = lm[8] - (lm[0] + lm[16]) * 0.5

    x = left_to_right - np.flipud(top_to_down) * [-1, 1]
    x /= np.hypot(*x)
    x *= np.hypot(*left_to_right)
    x *= scale
    y = np.flipud(x) * [-1, 1]

    c = ((lm[0] + lm[16]) * 0.5 + lm[8]) * 0.5

    x = np.flipud(x)
    y = np.flipud(y)
    c = np.flipud(c)

    return [c - x - y, c - x + y, c + x + y, c + x - y] + \
            interp(c - x - y, c + x - y) + \
            interp(c- x - y, c - x + y) + \
            interp(c - x + y, c + x + y) + \
            interp(c + x - y, c + x + y)




def image_deformation(image, image_ref, mode='rigid'):
    lm_face = get_landmark(image)
    lm_ref = get_landmark(image_ref)

    fixed_p = get_fixed_points(lm_face)
    # p = kalman_filter(lm_face[:17:2][:, ::-1])
    # p = [lm_face[3][::-1], lm_face[5][::-1], lm_face[9][::-1], lm_face[13][::-1], lm_face[15][::-1]]
    # q = [lm_ref[3][::-1], lm_ref[5][::-1], lm_ref[9][::-1], lm_ref[13][::-1], lm_ref[15][::-1]]
    # q = kalman_filter(lm_ref[:17:2][:, ::-1])
    p = kalman_filter(lm_face[:17][:, ::-1])
    q = kalman_filter(lm_ref[:17][:, ::-1])
    p = list(p) + fixed_p
    q = list(q) + fixed_p
    p = torch.from_numpy(np.array(p).copy()).to(device)
    q = torch.from_numpy(np.array(q).copy()).to(device)

    image = torch.from_numpy(np.array(image)).to(device)
    
    height, width, _ = image.shape
    gridX = torch.arange(width, dtype=torch.int16).to(device)
    gridY = torch.arange(height, dtype=torch.int16).to(device)
    vy, vx = torch.meshgrid(gridX, gridY)
    # !!! Pay attention !!!: the shape of returned tensors are different between numpy.meshgrid and torch.meshgrid
    vy, vx = vy.transpose(0, 1), vx.transpose(0, 1)
    
    if mode == 'affine':
        grid_new = mls_affine_deformation_pt(vy, vx, p, q, alpha=1)
    elif mode == 'similar':
        grid_new = mls_similarity_deformation_pt(vy, vx, p, q, alpha=1)
    elif mode == 'rigid':
        grid_new = mls_rigid_deformation_pt(vy, vx, p, q, alpha=1)
    else:
        raise ValueError(f'Wrong deformation mode {mode}.')


    # aug[vx.long(), vy.long()] = image[tuple(grid_new)]

    x = grid_new[0, :, :]
    y = grid_new[1, :, :]
            
    p1 = (x.floor().long(), y.floor().long())
    p2 = (p1[0], (p1[1] + 1).clamp_(0, width - 1))
    p3 = ((p1[0] + 1).clamp_(0, height - 1), p1[1])
    p4 = ((p1[0] + 1).clamp_(0, height - 1), (p1[1] + 1).clamp_(0, width - 1))

    fr1 = (p2[1] - y).unsqueeze(2) * image[(p1[0], p1[1])] + (y - p1[1]).unsqueeze(2) * image[(p2[0], p2[1])]
    fr2 = (p2[1] - y).unsqueeze(2) * image[(p3[0], p3[1])] + (y - p1[1]).unsqueeze(2) * image[(p4[0], p4[1])]
    aug = (p3[0] - x).unsqueeze(2) * fr1 + (x - p1[0]).unsqueeze(2) * fr2

    """
    fig, ax = plt.subplots()
    im = ax.imshow(aug.cpu().numpy())
    ax.scatter(lm_face[:, 0], lm_face[:, 1], c='b', s=2)
    ax.scatter(lm_ref[:, 0], lm_ref[:, 1], c='y', s=2)
    ax.scatter(np.array(fixed_p)[:, 1], np.array(fixed_p)[:, 0], c='r', s=5)
    
    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    im = plt.imshow(aug.cpu().numpy())
    plt.scatter(lm_face[:, 0], lm_face[:, 1], c='b', s=2)
    plt.scatter(lm_ref[:, 0], lm_ref[:, 1], c='y', s=2)
    plt.scatter(np.array(fixed_p)[:, 1], np.array(fixed_p)[:, 0], c='r', s=5)

    plt.savefig('/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video_res/fixed_points.png')

    """

    return Image.fromarray(np.uint8(aug.cpu().numpy()))



def image_deformation_local_warp(image, image_ref):
    lm_face = get_landmark(image)
    lm_ref = get_landmark(image_ref)


    end_point = lm_face[30]
    end_point_ref = lm_ref[30]

    image = np.array(image)
    for i in [3, 5, 9, 13, 15]:
        dist_target = np.linalg.norm(end_point - lm_face[i])
        dist_ref = np.linalg.norm(end_point_ref - lm_ref[i])

        if dist_ref < dist_target:
            image = local_traslation_warp(image, lm_face[i][::-1], end_point[::-1], radius=dist_target-dist_ref)

    return Image.fromarray(image)


def local_traslation_warp(image, start_point, end_point, radius):
    radius_square = math.pow(radius, 2)
    image_cp = image.copy()

    dist_se = math.pow(np.linalg.norm(end_point - start_point), 2)
    height, width, channel = image.shape
    for i in range(width):
        for j in range(height):
            # 计算该点是否在形变圆的范围之内
            # 优化，第一步，直接判断是会在（start_point[0], start_point[1])的矩阵框中
            if math.fabs(i - start_point[0]) > radius and math.fabs(j - start_point[1]) > radius:
                continue

            distance = (i - start_point[0]) * (i - start_point[0]) + (j - start_point[1]) * (j - start_point[1])

            if (distance < radius_square):
                # 计算出（i,j）坐标的原坐标
                # 计算公式中右边平方号里的部分
                ratio = (radius_square - distance) / (radius_square - distance + dist_se)
                ratio = ratio * ratio

                # 映射原位置
                new_x = i - ratio * (end_point[0] - start_point[0])
                new_y = j - ratio * (end_point[1] - start_point[1])

                new_x = new_x if new_x >=0 else 0
                new_x = new_x if new_x < width - 1 else width - 2
                new_y = new_y if new_y >= 0 else 0
                new_y = new_y if new_y < height - 1 else height - 2

                # 根据双线性插值法得到new_x, new_y的值
                image_cp[j, i] = bilinear_insert(image, new_x, new_y)
                
    return image_cp

# 双线性插值法
def bilinear_insert(image, new_x, new_y):
    w, h, c = image.shape
    if c == 3:
        x1 = int(new_x)
        x2 = x1 + 1
        y1 = int(new_y)
        y2 = y1 + 1

        part1 = image[y1, x1].astype(np.float) * (float(x2) - new_x) * (float(y2) - new_y)
        part2 = image[y1, x2].astype(np.float) * (new_x - float(x1)) * (float(y2) - new_y)
        part3 = image[y2, x1].astype(np.float) * (float(x2) - new_x) * (new_y - float(y1))
        part4 = image[y2, x2].astype(np.float) * (new_x - float(x1)) * (new_y - float(y1))

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.int8)


if __name__ == "__main__":
    # demo()
    # demo2()
    # demo3()
    demo_torch()

    # run_benckmark(i=0)
