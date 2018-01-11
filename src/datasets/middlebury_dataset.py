"""
Project:    m3n_fast_pd
File:       middlebury_dataset.py
Created by: louise
On:         10/01/18
At:         3:48 PM
"""

from __future__ import print_function
import os
import os.path
import re

from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np

from PIL import Image


def load_pfm(fn):
    """
    Function to load pfm file into PIL image.
    :param fn: str, file name
    :return:
    """
    in_file = open(fn)

    header = in_file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('File doesnt start with pf.')
    # Read file header for dimensions
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', in_file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Non valid PFM header.')

    # Read pgm scale data
    scale = float(in_file.readline().rstrip())
    if scale < 0:
        endian = '<'
    else:
        endian = '>'
    # Read pgm data
    data = np.fromfile(in_file, endian + 'f')
    if color:
        shape = (height, width, 3)
    else:
        shape = (height, width)
    im_np = np.reshape(data, shape)
    mask = np.ones(im_np.shape, dtype=float)
    mask[im_np==0] = 0.
    return Image.fromarray(im_np).convert("L"), Image.fromarray(mask).convert("L")


def indexer_middlebury(root, train=True):
    """

    :param root:
    :param train:
    :return:
    """
    items = os.listdir("/media/louise/data/datasets/middlebury/")

    newlist = []
    for names in items:
        if names.endswith("-perfect"):
            newlist.append(names)
    print(newlist)  # perfect folders

    images = []
    # Get disp 0, disp 1, im0, im1
    for folder in newlist:
        im0 = Image.open(os.path.join(root, folder, "im0.png")).convert("L")
        im1 = Image.open(os.path.join(root, folder, "im1.png")).convert("L")
        disp, mask = load_pfm(os.path.join(root, folder, "disp0.pfm"))

    images.append((im0, im1, disp, mask))

    return images


class Middlebury(Dataset):
    def __init__(self, root, indexer, transform=None, depth_transform=None):
        images = indexer(root)
        if len(images) == 0:
            raise (RuntimeError("No images in data set."))
        self.imgs = images
        self.transform = transform
        self.depth_transform = depth_transform

    def __getitem__(self, index):
        """
        __getitem__ overloading.
        :param index: int
        :return: 3-tuple of images.
        """
        # Read each image
        left_img = self.imgs[index][0]
        right_img = self.imgs[index][1]
        disp = self.imgs[index][2]
        mask = self.imgs[index][3]
        # Apply transforms to the images
        if self.transform is not None:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        if self.depth_transform is not None:
            disp = self.depth_transform(disp)
            mask = self.depth_transform(mask)
        disp = transforms.ToTensor()(disp)
        mask = transforms.ToTensor()(mask)

        # Return 3-tuple
        return left_img, right_img, disp, mask

    def __len__(self):
        return len(self.imgs)
