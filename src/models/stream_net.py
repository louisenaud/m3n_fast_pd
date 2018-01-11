"""
Project:    m3n_fast_pd
File:       net_2ch_2stream.py
Created by: louise
On:         08/01/18
At:         5:15 PM
"""
import torch
import torch.nn as nn
import torch.functional as F


class NetStream(nn.Module):
    """
    Return score for two img_patch of dimensions:
        img_patch: B x C x H x W x Hp x Wp
        score: B x H x W
    """

    def __init__(self, num_input_channel=1):
        super(NetStream, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv4 = nn.Conv2d(10, 20, kernel_size=5)

    def forward(self, img):
        o = self.conv1(img)
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = self.conv2(o)
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = self.conv3(o)
        o = F.relu(o)
        o = self.conv4(o)
        o = F.relu(o)
        return o.view(o.size(0), -1)