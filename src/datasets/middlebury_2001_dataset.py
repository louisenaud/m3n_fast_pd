"""
Project:    m3n_fast_pd
File:       middlebury_2001_dataset.py
Created by: louise
On:         11/01/18
At:         2:13 PM
"""
from __future__ import print_function
import os
import os.path
import re

from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np

from PIL import Image


def load_pgm(filename, byteorder='>'):
    """
    Source mainly from: https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
    Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer_im = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer_im).groups()
    except AttributeError:
        raise ValueError("Does not recognize PGM file: '%s'" % filename)
    disp = np.frombuffer(buffer_im, dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                         count=int(width)*int(height),
                         offset=len(header)).reshape((int(height), int(width)))
    disp /= 8.
    mask = np.ones(disp.shape, dtype=float)
    mask[disp == 0.] = 0.
    return Image.fromarray(disp).convert("L").transpose(Image.FLIP_TOP_BOTTOM), \
           Image.fromarray(mask).convert("L").transpose(Image.FLIP_TOP_BOTTOM)


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
    im_np[im_np==0] = -1
    im_np[im_np=='Inf'] = 255
    mask = np.ones(im_np.shape, dtype=float)
    mask[im_np==0] = 0.
    return Image.fromarray(im_np).convert("L").transpose(Image.FLIP_TOP_BOTTOM), \
           Image.fromarray(mask).convert("L").transpose(Image.FLIP_TOP_BOTTOM)


def indexer_middlebury_2001(root, train=True):
    """

    :param root:
    :param train:
    :return:
    """
    basenames = ['barn1', 'barn2', 'bull', 'poster', 'sawtooh', 'tsukuba', 'venus']
    images = []
    for basename in basenames:
        print(os.path.join(root, basename + "1.ppm"))
        im0 = Image.open(os.path.join(root, basename + "1.ppm")).convert("L")
        im1 = Image.open(os.path.join(root, basename + "2.ppm")).convert("L")
        disp, mask = load_pgm(os.path.join(root, basename + "GT.pgm"))
        images.append((im0, im1, disp, mask))

    return images


class Middlebury2001(Dataset):
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
        :return: 4-tuple of images.
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
