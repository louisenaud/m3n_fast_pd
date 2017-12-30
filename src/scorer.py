
# Pytorch module
# Input: img_0_patch and img_1_patch B x C x H x W x Ph xPw
# Ouput: img_score: B x H x W
# model: Zncc on 5x5 grid or weighted L1 wrt central pixel (this used to be efficient in middleburry).


import torch
import torch.nn as nn
from torch.autograd import Variable
from interpolator import Interpolator


class Scorer(nn.Module):
    """ Return score for two img_patch
        img_patch: B x C x H x W x Hp x Wp
        score: B x H x W
    """

    def __init__(self, parameters=None, num_input_channel=1):
        super(Scorer, self).__init__()

        if not parameters:
            self.alpha = torch.FloatTensor(num_input_channel*[0.])
        else:
            self.alpha = torch.FloatTensor([parameters['alpha']])

        self.alpha = nn.Parameter(self.alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3))

    def paramterers_constraint(self):
        """ Apply box constraint on parameters"""

        self.alpha.data.clamp(0., 1000.)

    def forward(self, img0_patch, img1_patch):

        score = torch.abs(img0_patch-img1_patch)
        score = torch.mean(torch.mean(score, dim=5), dim=4)

        score = torch.mul(self.alpha, score)
        score = torch.mean(score, dim=1)

        return score
