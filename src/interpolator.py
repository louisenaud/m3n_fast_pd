# Pytorch module
# Input: img: B x C x H x W and x BxHxW
# Ouput: img_inter B x C x H x W
# model: extract pixel at position grid + x from input img


import torch
import torch.nn as nn
from torch.autograd import Variable


class Interpolator(nn.Module):
    """
    Return img_interp where:
        img_interp(i, j) = img(i+x(i,j,0), j+x(i,j,1))
    """

    def __init__(self, ):
        super(Interpolator, self).__init__()

    @staticmethod
    def make_grid(H, W):
        """
        Return a grid of dimension 1 x H x W x 2
            grid lives in [0, H-1] x [0, W-1]
            grid[0, 10, 5, 0] = 10
            grid[0, 10, 5, 1] = 5
        :param H: int, height of grid
        :param W: int, width of grid
        :return Pytorch Variable, [1xHxWx2]
        """

        grid_h = torch.arange(start=0, end=H, step=1)
        grid_h = grid_h.unsqueeze(0).unsqueeze(2)
        grid_h = grid_h.expand(1, H, W)

        grid_w = torch.arange(start=0, end=W, step=1)
        grid_w = grid_w.unsqueeze(0).unsqueeze(1).expand(1, H, W)

        return Variable(torch.stack([grid_w, grid_h], dim=3))

    def make_pytorch_grid(self, H, W):
        """
        Create Grid of dimension 1 x H x W x 2 in Pytorch.
        grid lives in [-1., 1.] x [-1., 1.]
        grid[0, 0, 0, 0] = -1.
        grid[0, 0, W-1, 1] = 1.

        :param H: int, height of grid.
        :param W: int, width of grif.
        :return: Pytorch Variable, [1xHxWx2]
        """

        grid = self.make_grid(H, W)
        grid[:, :, :, 0] = grid[:, :, :, 0] / float(W-1)
        grid[:, :, :, 1] = grid[:, :, :, 1] / float(H-1)
        grid = 2. * grid - 1.

        return grid

    def interp_grid(self, x):
        """
        Interpolate the grid.
        :param x:
        :return:
        """
        B, H, W = x.size()[0:3]

        pytorch_grid = self.make_pytorch_grid(H, W).type_as(x)

        new_grid = Variable(torch.DoubleTensor(B, H, W, 2)).type_as(x)
        for b in range(B):
            new_grid[b, :, :, 0] = pytorch_grid[:, :, :, 0] + 2. * x[b, :, :, 0] / float(W-1)
            new_grid[b, :, :, 1] = pytorch_grid[:, :, :, 1] + 2. * x[b, :, :, 1] / float(H-1)

        return new_grid

    def forward(self, img, x):
        """
        Interpolates grid.
        :param img:
        :param x:
        :return:
        """
        interp_grid = self.interp_grid(x)

        return nn.functional.grid_sample(img, interp_grid, mode='nearest', padding_mode='border')


