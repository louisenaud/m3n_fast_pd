"""
Project:    m3n_fast_pd
File:       energy.py
Created by: louise
"""

# Pytorch module
# Input: img0 and img1: B x C x H x W and x BxHxW
# Ouput: nrg B
# model: compute energy for triplet (img0, img1, x)

import torch
import torch.nn as nn


class Energy(nn.Module):

    def __init__(self, unary, pairwise):
        super(Energy, self).__init__()

        self.unary = unary
        self.pairwise = pairwise

    def parameters_constraint(self):
        """
        Apply constraint on parameters
        """

        self.unary.parameters_constraint()
        self.pairwise.parameters_constraint()

    def forward(self, img0, img1, x):
        """

        :param img0:
        :param img1:
        :param x:
        :return:
        """
        unary = self.unary.forward(img0, img1, x)
        unary_cost = torch.sum(torch.sum(unary, dim=2), dim=1)

        pairwise = self.pairwise.forward(img0, x)
        pairwise_cost = torch.sum(torch.sum(torch.sum(pairwise, dim=3), dim=2), dim=1)

        return unary_cost + pairwise_cost

