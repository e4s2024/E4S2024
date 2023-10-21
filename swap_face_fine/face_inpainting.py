
import torch
from PIL import Image
import numpy as np
import cv2

# import sys
# sys.path.insert(0, "/apdcephfs_cq2/share_1290939/branchwang/projects/GCFSR")

from .gcfsr_arch import FaceInpaintingArch


model_path = './pretrained/inpainting/net_g_50000.pth'
model = FaceInpaintingArch(out_size=256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load(model_path)['params_ema'], strict=True)
model.eval()
model = model.to(device)

def inpainting(face_img, mask):
    img = face_img.resize((256, 256))
    img = np.array(img) / 255.

    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)
    
    mask = cv2.resize(mask.astype(np.float), (256, 256))
    mask = (mask[None, :, :] > 0)
    mask = torch.from_numpy(mask).float()
    mask = mask.unsqueeze(0).to(device)

    img_lq = img * (1 - mask)
    img_lq = torch.cat([img_lq, mask], dim=1)

    in_size = torch.sum(mask) / (256 * 256)
    in_size = in_size.cpu()
    cond = torch.from_numpy(np.array([in_size], dtype=np.float32))
    cond = cond.to(device)

    with torch.no_grad():
        output, _ = model(img_lq, cond)
        output = output.clamp(0., 1.)
        output = img * (1 - mask) + output * mask

    output = output.data.squeeze().float().cpu().numpy()
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    output = Image.fromarray(output)

    return output