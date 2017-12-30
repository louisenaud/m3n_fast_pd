
import torch
import torch.nn as nn


class Margin(nn.Module):

    def __init__(self):
        super(Margin, self).__init__()

    def forward(self, x_ref, x):

        margin_cost = torch.clamp(torch.abs(x_ref - x), 0, 1)

        sum_margin_cost = torch.sum(torch.sum(margin_cost, dim=2), dim=1)

        return sum_margin_cost
