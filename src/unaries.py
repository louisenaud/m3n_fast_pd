
# Pytorch module
# Input: img_0, img_1 BxCxHxW and x BxHxW
# Ouput: cost BxHxW
# model: each element of the unary term is computed on a patch
# Sub-module:
#   - patch extractor: img: B x C x H x W and x BxHxW --> img_patch: B x C x H x W x Ph xPw (make use of interpolator)
#   - scorer: img_0_patch and img_1_patch --> img_score: B x 1 x H x W or B x H x W



import torch
import torch.nn as nn
from patch_extractor import PatchExtractor


class Unary(nn.Module):
    """ Return score for two img_patch
        img_patch: B x C x H x W x Hp x Wp
        x: B x H x W x 2
        score: B x H x W
    """

    def __init__(self, scorer, patch_shape=(3, 3)):
        super(Unary, self).__init__()

        self.score = scorer
        self.patch_shape = patch_shape
        self.patch_extractor = PatchExtractor(self.patch_shape)

    def paramterers_constraint(self):
        """ Apply constraint on parameters"""

        self.score.paramterers_constraint()

    def forward(self, img0, img1, x):

        # Create patches from img0 and img1
        img0_patch = self.patch_extractor.forward(img0)
        img1_patch = self.patch_extractor.forward(img1, x)

        # Compute score
        unary = self.score(img0_patch, img1_patch)

        return unary
