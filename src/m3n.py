"""
Project:    m3n_fast_pd
File:       m3n.py
Created by: louise
"""

# Pytorch module
# Input: x_gt, x_min BxHxW and img_0, img_1 BxCxHxW
# Ouput: loss B
# model: E(x_gt) - E_bar(x_min)

import torch.nn as nn


class M3N(nn.Module):
    def __init__(self, energy, margin):
        super(M3N, self).__init__()

        self.energy = energy
        self.margin = margin

    def paramterers_constraint(self):
        """
        Apply constraint on parameters
        """

        self.energy.paramterers_constraint()

    def forward(self, img0, img1, x_gt, x_min):
        # Get energy of x_gt and x_min
        nrg_gt = self.energy.forward(img0, img1, x_gt)
        nrg_min = self.energy.forward(img0, img1, x_min)

        # Add margin cost
        nrg_min_with_margin = nrg_min - self.margin(x_gt, x_min)

        return nrg_gt - nrg_min_with_margin
