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
import os.path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy

import png

from PIL import Image


def read_disparity(img_path):
    """
    Reads a 16-bit depth png
    :param img_path:
    :return:
    """
    r = png.Reader(filename=img_path)
    row_count, column_count, pngdata, meta = r.asDirect()
    image_2d = numpy.vstack(itertools.imap(numpy.uint16, pngdata))
    im2 = image_2d.astype(float)
    im2 /= 255.
    mask = numpy.ones(im2.shape, dtype=numpy.float)
    mask[im2 == 0.] = 0.
    return Image.fromarray(im2).convert("L"), \
           Image.fromarray(mask).convert("L")


def indexer_KITTI(root, train=True):
    """

    :param root:
    :param train:
    :return:
    """
    # TODO(louise): add training / testing configuration
    stereo_l = os.path.join(root, 'image_0')
    stereo_r = os.path.join(root, 'image_1')
    disparity = os.path.join(root, 'disp_noc')
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
        # Get path of right, left and disparity images
        left_path, right_path, disp_path = self.imgs[index]
        # Read each image
        left_img = Image.open(left_path).convert("L")
        right_img = Image.open(right_path).convert("L")
        disp, mask = read_disparity(disp_path)
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
