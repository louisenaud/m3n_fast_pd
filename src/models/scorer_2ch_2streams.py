"""
Project:    m3n_fast_pd
File:       scorer_2ch_2streams.py
Created by: louise
On:         08/01/18
At:         4:30 PM
"""


import torch
import torch.nn as nn
import torch.functional as F


class Scorer2Ch2Stream(nn.Module):
    """
    Return score for two img_patch of dimensions:
        img_patch: B x C x H x W x Hp x Wp
        score: B x H x W
    """

    def __init__(self, parameters=None, num_input_channel=1):
        super(Scorer2Ch2Stream, self).__init__()

        if not parameters:
            self.alpha = torch.FloatTensor(num_input_channel*[1.])
        else:
            self.alpha = torch.FloatTensor([parameters['alpha']])

        self.alpha = nn.Parameter(self.alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv4 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def parameters_constraint(self):
        """
        Apply box constraint on parameters
        :return
        """

        self.alpha.data.clamp(0., 1000.)

    def stream(self, img):
        B, C, H, W = img.size()
        o = self.conv1(img)
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = self.conv2(o)
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = self.conv3(o)
        o = F.relu(o)
        o = self.conv4(o)
        o = F.relu(o)
        return o.view(o.size(0), -1)

    def forward(self, img0_patch, img1_patch):
        """
        Computes the score for 2 image patches.
        :param img0_patch, B x C x H x W x Hp x Wp
        :param img1_patch, B x C x H x W x Hp x Wp
        :return: score, B x H x W
        """

        score = 0.0
        o_fovea = stream(F.avg_pool2d(input, 2, 2), 'fovea')
        o_retina = stream(F.pad(input, (-16,) * 4), 'retina')
        o = linear(torch.cat([o_fovea, o_retina], dim=1), params, 'fc0')
        return

        return linear(F.relu(o))
#####################   2ch   #####################

def deepcompare_2ch(input, params):
    o = conv2d(input, params, 'conv0', stride=3)
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, 'conv1')
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, 'conv2')
    o = F.relu(o).view(o.size(0), -1)
    return linear(o, params, 'fc')


#####################   2ch2stream   #####################

def deepcompare_2ch2stream(input, params):

    def stream(input, name):
        o = conv2d(input, params, name + '.conv0')
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + '.conv1')
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + '.conv2')
        o = F.relu(o)
        o = conv2d(o, params, name + '.conv3')
        o = F.relu(o)
        return o.view(o.size(0), -1)

    o_fovea = stream(F.avg_pool2d(input, 2, 2), 'fovea')
    o_retina = stream(F.pad(input, (-16,) * 4), 'retina')
    o = linear(torch.cat([o_fovea, o_retina], dim=1), params, 'fc0')
    return linear(F.relu(o), params, 'fc1')