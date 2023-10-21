"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, specific_ids: list = None):
    """找出一个文件夹下面的所有图片文件，
    Args:
        dir (str): the directory to be searched
        specific_ids (list[str]): optional, indicate the specific ids
    Returns:
        list: 所有的图片列表
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if (specific_ids is not None) and (fname not in specific_ids):
                    continue  # skip this id
                images.append(path)
    return images
