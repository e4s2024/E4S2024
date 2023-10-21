import numpy as np
import torch
import os

from utils import torch_utils


def fill_hole(radius=5, eye_line=218):
    # hole = hole_mask.copy()
    mask = np.load('/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/mask.npy')
    hole = np.load('/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/hole.npy')
    k = 1
    # while k < 500:
    while hole.any():
        k += 1
        mask_new = mask.copy()
        n, m = len(mask), len(mask[0])
        for i in range(n):
            for j in range(m):
                if hole[i][j]:
                    if i < eye_line:
                        mask_new[i][j] = 6
                        hole[i][j] = False
                        continue
                    # neighbors = mask[max(0, i - radius * 10): min(i + radius * 10, n), max(0, j - radius): min(j + radius, m)]
                    neighbors = mask[max(0, i - radius): min(i + radius, n - 1), j-2: j+2]
                    neighbors = list(neighbors[neighbors != 1])
                    try:
                        mask_new[i][j] = max(neighbors, key=neighbors.count)
                        hole[i][j] = False
                    except:
                        print(neighbors)
                        print(i, j)
                        # mask_new[i][j] = 6
                        # hole[i][j] = False
                        # pass
        mask = mask_new
    # mask[mask == 1] = 0
    # mask[hole] = 1
    return mask


hole_map = fill_hole()

hole_one_hot = torch_utils.labelMap2OneHot(torch.from_numpy(hole_map).unsqueeze(0).unsqueeze(0).long(), num_cls=12)
torch_utils.tensor2map(hole_one_hot[0]).save(os.path.join("hole_map.png"))