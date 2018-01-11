"""
Project:    m3n_fast_pd
File:       siamese_net.py
Created by: louise
On:         08/01/18
At:         5:55 PM
"""
import torch
import torch.nn as nn
import torch.functional as F


class UnitSiamese(nn.Module):
    """
    Branch of a siamese network.
    """
    def __init__(self, num_input_channel=1):
        super(UnitSiamese, self).__init__()

        self.conv1 = nn.Conv2d(1, 96, kernel_size=7, stride=3)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=5)
        self.conv3 = nn.Conv2d(192, 256, kernel_size=3)

    def forward(self, patch):
        o = self.conv1(patch)
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = self.conv2(o)
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = self.conv3(o)
        o = F.relu(o)
        return o.view(o.size(0), -1)


class Siamese(nn.Module):
    """
    Siamese network that outputs features.
    """
    def __init__(self, num_input_channel=1):
        super(Siamese, self).__init__()
        self.siam1 = UnitSiamese()
        self.linear1 = nn.Linear(256 * 2, 256 * 2)

    def forward_once(self, x):
        output = self.siam1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, patch1, patch2):
        o1 = self.siam1.forward_once(patch1)
        o2 = self.siam1.forward_once(patch2)
        o = torch.cat([o1, o2], dim=1)
        o = self.linear1.forward(o)

        return o


class SiameseScorer(nn.Module):
    """
    Return score for two img_patch of dimensions:
        img_patch: B x C x H x W x Hp x Wp
        score:     B x H x W
    """
    def __init__(self, num_input_channel=1):
        super(SiameseScorer, self).__init__()

        self.siamese = Siamese()
        self.linear = nn.Linear(2*256, 1)

    def forward(self, patch1, patch2):
        o = self.siamese.forward(patch1, patch2)
        o = F.relu(o)
        score = self.linear(o)
        return score
