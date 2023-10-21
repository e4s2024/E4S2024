import time
import logging
import sys
import os
import shutil
import numpy as np


def early_stop(test_acc, stop_th, printf=print):
    if stop_th <= 1:
        return

    if len(test_acc) < stop_th:
        return

    accs = test_acc[-stop_th:]
    deltas = [accs[i] - accs[i + 1] for i in range(len(accs) - 1)]

    if np.min(deltas) > 0:
        printf('early stop')
        printf(accs)
        raise TimeoutError


def denorm_img(timgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    simage = (timgs * timgs.new(std).unsqueeze(0).unsqueeze(2).unsqueeze(2) + timgs.new(
        mean).unsqueeze(0).unsqueeze(2).unsqueeze(2)).clamp(0, 1)
    return simage


def get_nimage(timage, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return denorm_img(timage, mean=mean, std=std).permute(0, 2, 3, 1).data.cpu().numpy()


def copy_to(source, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy(source, target)


class Timer(object):
    def __init__(self, title='this block', printf=print, unit=1):
        self.title = title
        self.printf = printf
        self.unit = unit

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.printf is not None:
            self.printf(f'Time for {self.title}: {(time.time() - self.t0) / self.unit:.4f} (*{self.unit}) seconds')


def init_log(exp_name):
    save_path = os.path.join('output', exp_name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'checks'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'train_viz'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'test_viz'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'val_viz'), exist_ok=True)

    log_fname = f'{save_path}/{exp_name}.log'

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    handler = logging.FileHandler(log_fname)
    handler.setFormatter(logging.Formatter('%(asctime)s: |%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    mlog('')
    mlog(exp_name)
    mlog('')


def mlog(msg=''):
    logging.getLogger().info(str(msg))


def set_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def occupy(g=10):
    if g <= 0:
        return
    import torch
    for i in range(torch.cuda.device_count()):
        try:
            x = torch.cuda.FloatTensor(256, 1024, int(1000 * g), device=i)
            del x
        except Exception:
            print('OOM')
