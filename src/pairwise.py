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

    def paramterers_constraint(self):

        self.weights.paramterers_constraint()
        self.distance.paramterers_constraint()

    def gradient_operator(self, x):

        B, H, W = x.size()[0:3]

        g_v = torch.zeros_like(x)
        g_v[:, 0:H-1, :] = x[:, 0:H-1, :] - x[:, 1:, :]

        g_h = torch.zeros_like(x)
        g_h[:, :, 0:W-1] = x[:, :, 0:W-1] - x[:, :, 1:]

        return torch.stack([g_v, g_h], dim=3)

    def forward(self, img0, x):

        w = self.weights.forward(img0)
        gx = self.gradient_operator(x)
        d = self.distance(gx)

        return torch.mul(w, d)

