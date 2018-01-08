"""
Project:    m3n_fast_pd
File:       patch_extractor.py
Created by: louise
"""

# Pytorch module
# Input: img: B x C x H x W and x BxHxW
# Ouput: img_patch: B x C x H x W x Ph xPw
# model: extract patch at centered in position x from input img (make use of interpolator)


import torch
import torch.nn as nn
from torch.autograd import Variable
from interpolator import Interpolator


class PatchExtractor(nn.Module):
    """
    Return patches centered around x for all image pixels
    """

    def __init__(self, patch_shape=(3, 3)):
        super(PatchExtractor, self).__init__()

        self.patch_shape = patch_shape
        self.pi = patch_shape[0] - int((patch_shape[0] - 1) / 2)
        self.pj = patch_shape[1] - int((patch_shape[1] - 1) / 2)

    def forward(self, img, x=None):
        """
        Extract patches of size patch_shape.
        :param img: Pytorch Variable [BxCxHxW]
        :param x: Pytorch Variable [BxHxW]
        :return: Pytorch Variable [BxCxHxWx Hpatch x Wpatch]
        """
        # Get size
        B, C, H, W = img.size()
        Hp, Wp = self.patch_shape

        # Create interpolator
        interpolator = Interpolator()

        # Create patches
        img_patch = Variable(torch.DoubleTensor(B, C, H, W, Hp, Wp)).type_as(img)

        for i in range(Hp):
            for j in range(Wp):

                # Create x_disp
                flow = Variable(torch.FloatTensor(B, H, W, 2)).type_as(img)
                if x is None:
                    flow[:, :, :, 0] = float(j - self.pj)
                    flow[:, :, :, 1] = float(i - self.pi)
                else:
                    flow[:, :, :, 0] = -x + float(j - self.pj)
                    flow[:, :, :, 1] = float(i - self.pi)

                # Apply
                img_patch[:, :, :, :, i, j] = interpolator.forward(img, flow)

        return img_patch
