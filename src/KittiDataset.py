"""
Project:    m3n_fast_pd
File:       KittiDataset.py
Created by: louise
On:         29/12/17
At:         6:05 PM
"""
from __future__ import print_function
import itertools
import os
from os.path import abspath

import numpy as np
from PIL import Image

from torch.utils.data.dataset import Dataset

import torch
import os
import os.path

import imageio
from PIL import Image


def read_disparity(img_path):
    im2 = imageio.imread(img_path)
    im2 = im2.astype(float)
    return torch.from_numpy(im2)


def is_image_file(filename):
    """

    :param filename:
    :return:
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_pfm_file(filename):
    """

    :param filename: str
    :return: bool
    """
    return filename.endswith('.pfm')


def indexer_KITTI(root, train=True):
    """

    :param root:
    :param train:
    :return:
    """
    #status = 'training' if train else 'testing'
    stereo_l = os.path.join(root, 'image_0')
    print(stereo_l)
    stereo_r = os.path.join(root, 'image_1')
    disparity = os.path.join(root, 'disp_refl_noc')
    images = []

    for l, r, d in itertools.izip(sorted(os.listdir(stereo_l)),
                                  sorted(os.listdir(stereo_r)),
                                  sorted(os.listdir(disparity))):

        img_l = os.path.abspath(os.path.join(stereo_l, l))
        img_r = os.path.abspath(os.path.join(stereo_r, r))
        disp = os.path.abspath(os.path.join(disparity, d))

        images.append((img_l, img_r, disp))

    return images


class KITTI(Dataset):
    def __init__(self, root, indexer, transform=None, depth_transform=None):
        images = indexer(root)
        if len(images) == 0:
            raise (RuntimeError("Found 0 images in folders."))
        self.imgs = images
        self.transform = transform
        self.depth_transform = depth_transform

    def __getitem__(self, index):
        """
        __getitem__ overloading.
        :param index: int
        :return: 3-tuple of images.
        """
        # Get path of right, left and disparity images
        left_path, right_path, disp_path = self.imgs[index]
        # Read each image
        left_img = Image.open(left_path).convert("L")
        right_img = Image.open(right_path).convert("L")
        disp = read_disparity(disp_path)
        # Apply transforms to the images
        if self.transform is not None:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        if self.depth_transform is not None:
            disp = self.depth_transform(disp)

        # Return 3-tuple
        return left_img, right_img, disp

    def __len__(self):
        return len(self.imgs)
