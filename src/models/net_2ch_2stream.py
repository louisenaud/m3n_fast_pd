"""
Project:    m3n_fast_pd
File:       net_2ch_2stream.py
Created by: louise
On:         08/01/18
At:         5:24 PM
"""
import torch
import torch.nn as nn
import torch.functional as F
from stream_net import NetStream


class Net2ch2str(nn.Module):
    """
    Return score for two img_patch of dimensions:
        img_patch: B x C x H x W x Hp x Wp
        score: B x H x W
    """

    def __init__(self, num_input_channel=1):
        super(Net2ch2str, self).__init__()

        self.stream1 = NetStream()
        self.stream2 = NetStream()
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, batch):
        o_fovea = self.stream1(F.avg_pool2d(batch, 2, 2), 'fovea')
        o_retina = self.stream2(F.pad(batch, (-16,) * 4), 'retina')

        return torch.cat([o_fovea, o_retina], dim=1)
