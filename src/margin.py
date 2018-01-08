"""
Project:    m3n_fast_pd
File:       margin.py
Created by: louise
"""
import torch
import torch.nn as nn


class Margin(nn.Module):

    def __init__(self):
        super(Margin, self).__init__()

    def forward(self, x_ref, x, mask=None):
        """
        Compute margin in energy.
        :param x_ref: Pytorch Variable, [BxHxW]
        :param x: Pytorch Variable, [BxHxW]
        :param mask: Pytorch Variable, [BxHxW]
        :return: Pytorch Variable, [BxHxW]
        """
        margin_cost = torch.clamp(torch.abs(x_ref - x), 0, 1)

        if mask is not None:
            margin_cost = torch.mul(mask, margin_cost)

        return margin_cost
