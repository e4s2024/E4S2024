import torch
from .trainer import Trainer


def get_face_blender(args):
    model = Trainer(args)
    return model
