
import os
import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
# from skimage.color import rgb2gray, gray2rgb

from swap_face_fine.MISF.src.networks import InpaintGenerator
from swap_face_fine.MISF.src.config import Config
from utils.morphology import dilation



def load_config():

    ckpt_path = "/apdcephfs_cq2/share_1290939/branchwang/projects/misf/checkpoints"
    config_path = os.path.join(ckpt_path, 'config.yml')

    # load config file
    config = Config(config_path)

    config.MODE = 2

    return config






def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t

def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).contiguous()
    return img.int()





def inpainting_face(img, mask):
    # img = np.array(img)
    # img_gray = rgb2gray(img)
    
    images = to_tensor(np.uint8(img)).unsqueeze(0)
    # img_gray = to_tensor(img_gray).unsqueeze(0)
    mask = mask[:, :, None]
    mask = np.repeat(mask, 3, axis=-1)
    masks = to_tensor(np.uint8(mask)).unsqueeze(0)

    # radius = 5
    # masks = dilation(masks, torch.ones(2 * radius + 1, 2 * radius + 1, device=masks.device), engine='convolution')

    config = load_config()

    gen_weights_path = "/apdcephfs_cq2/share_1290939/branchwang/projects/misf/checkpoints/celebA_InpaintingModel_gen.pth"

    inpaint_model = InpaintGenerator(config).to(config.DEVICE)

    if torch.cuda.is_available():
        data = torch.load(gen_weights_path)
    else:
        data = torch.load(gen_weights_path, map_location=lambda storage, loc: storage)

    inpaint_model.load_state_dict(data['generator'])

    inpaint_model.eval()

    images_masked = images * (1 - masks)
    inputs = torch.cat((images_masked, masks[:, :1, :, :]), dim=1)
    outputs = inpaint_model(inputs) 
    outputs_merged = (outputs * masks) + images * (1 - masks)

    res = postprocess(outputs_merged)[0].cpu().numpy()
    return res