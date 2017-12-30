"""
Project:    m3n_fast_pd
File:       KittiDataset.py
Created by: louise
On:         29/12/17
At:         6:05 PM
"""
from __future__ import print_function
import os
from os.path import abspath

import numpy as np
from PIL import Image

from torch.utils.data.dataset import Dataset

import torch
import torch.utils.data as data
from torchvision import transforms
import os
import os.path
from scipy.ndimage import imread
import glob
import sys
import re
import random
import math
from PIL import Image
import numbers

import re
import sys
from PIL import Image, ImageOps


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def write_pfm_image(file_name, img):
    """
    Write image in the PFM format.
    :param file_name: str, file name for output file
    :param img: Pytorch Variable, to write in output file
    :return:
    """

    file_ = open(file_name, 'wb')

    if img.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

        img = np.flipud(img)

    if len(img.shape) == 3 and img.shape[2] == 3:  # color image
        color = True
    elif len(img.shape) == 2 or len(img.shape) == 3 and img.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file_.write('PF\n' if color else 'Pf\n')
    file_.write('%d %d\n' % (img.shape[1], img.shape[0]))
    image.tofile(file_)


def read_pfm_image(img_path):
    """
    read a pfm img
    :param img_path: str, path to pfm image.
    :return: H x W x C numpy array
    """
    with open(img_path, 'rb') as file:
        header = file.readline().rstrip().decode(encoding='utf-8')
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encoding='utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip().decode(encoding='utf-8'))
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width, 1)

        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data


def read_color_image(path):
    """

    :param path:
    :return:
    """
    return Image.open(path).convert('RGB')


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
    status = 'TRAIN' if train else 'TEST'
    stereo = os.path.join(root, 'frames_cleanpass', status)
    disparity = os.path.join(root, 'disparity', status)
    images = []

    for type in ['A', 'B', 'C']:
        for scene in sorted(os.listdir(os.path.join(stereo, type))):
            for fname in sorted(os.listdir(os.path.join(stereo, type, scene, 'left'))):
                if is_image_file(fname):
                    left_path, right_path = os.path.join(stereo, type, scene, 'left', fname), os.path.join(stereo, type, scene,
                                                                                                  'right', fname)
                    disp_path = os.path.join(disparity, type, scene, 'left', fname[:-3]+'pfm')
                    images.append((left_path, right_path, disp_path))

    return images


class KITTI(data.Dataset):
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
        left_img = read_color_image(left_path)
        right_img = read_color_image(right_path)
        disparity = read_pfm_image(disp_path)
        # Apply transforms to the images
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)
        disparity = self.depth_transform(disparity)
        # Return 3-tuple
        return left_img, right_img, disparity

    def __len__(self):
        return len(self.imgs)


def create_KITTI_filelist(img_path):
    abs_path = abspath('.')
    path_img = os.path.join(abs_path, img_path)
    filelist = os.listdir(path_img)
    for fichier in filelist[:]:  # filelist[:] makes a copy of filelist.
        if not (fichier.endswith(".png") or not (fichier.endswith(".jpg"))):
            filelist.remove(fichier)

    return filelist


class KITTI_Images(Dataset):
    """
    Dataset for noised images and GT.
    """

    def __init__(self, img_path, transform=None):
        self.filelist = create_KITTI_filelist(img_path)
        self.transform = transform

    def __getitem__(self, index):
        print("File : ", self.filelist[index])
        img = Image.open(self.filelist[index])
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        return img

    def __len__(self):
        return len(self.filelist)