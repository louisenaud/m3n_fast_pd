"""
Project:    m3n_fast_pd
File:       pairwise.py
Created by: louise
"""
# Pytorch module
# Input: img0 and img1: B x C x H x W and x BxHxW
# Ouput: nrg B
# model: compute energy for triplet (img0, img1, x)

import torch
import torch.nn as nn
from patch_extractor import PatchExtractor


class Pairwise(nn.Module):

    def __init__(self, weights, distance):
        super(Pairwise, self).__init__()

        self.weights = weights
        self.distance = distance

    def parameters_constraint(self):
        self.weights.parameters_constraint()
        self.distance.parameters_constraint()

    def gradient_operator(self, x):
        """
        Computes gradient of an image.
        :param x: PyTorch Variable
        :return: PyTorch Variable
        """
        B, H, W = x.size()[0:3]

        g_v = torch.zeros_like(x)
        g_v[:, 0:H-1, :] = x[:, 0:H-1, :] - x[:, 1:, :]

        g_h = torch.zeros_like(x)
        g_h[:, :, 0:W-1] = x[:, :, 0:W-1] - x[:, :, 1:]

        return torch.stack([g_v, g_h], dim=3)

    def gradient_mask(self, mask):

        B, H, W = mask.size()[0:3]

        g_v = torch.zeros_like(mask)
        g_v[:, 0:H - 1, :] = torch.mul(mask[:, 0:H - 1, :], mask[:, 1:, :])

        g_h = torch.zeros_like(mask)
        g_h[:, :, 0:W - 1] = torch.mul(mask[:, :, 0:W - 1], mask[:, :, 1:])

        return torch.stack([g_v, g_h], dim=3)

    def forward(self, img0, x, mask=None):
        """

        :param img0: PyTorch Variable
        :param x: PyTorch Variable
        :return: PyTorch Variable
        """
        w = self.weights.forward(img0)
        gx = self.gradient_operator(x)
        d = self.distance(gx)
        cost = torch.mul(w, d)

        # Apply mask
        if mask is not None:
            g_mask = self.gradient_mask(mask)
            cost = torch.mul(g_mask, cost)

        return cost

