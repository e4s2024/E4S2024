import numpy as np
from PIL import Image
from swap_face_fine.GMA.evaluate_single import estimate_flow
import torch

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


img1 = load_image('/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video_res/swap_elon_to_874/intermediate_results/imgs/D_0001.png')
img2 = load_image('/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video_res/swap_elon_to_874/intermediate_results/imgs/D_0002.png')

flow = estimate_flow(img1, img2)

print(flow)