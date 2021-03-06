"""
Project:    m3n_fast_pd
File:       weights.py
Created by: louise
"""
import torch
import torch.nn as nn


class Weights(nn.Module):
    """
    Compute the regularization weights from an image:
        w = \lambda_cst + \lambda_apt * exp(-inv_sigma * |conv_3x3(img)| ^ \alpha)
        img: BxCxHxW
        w = BxHxWx2
    """

    def __init__(self, parameters=None, num_input_channel=1):
        super(Weights, self).__init__()

        if not parameters:
            self.lambda_cst = nn.Parameter(torch.FloatTensor([0.01]))
            self.lambda_apt = nn.Parameter(torch.FloatTensor([0.1]))
            self.inv_sigma = nn.Parameter(torch.FloatTensor([1.]))  # 20. previously
            self.alpha = nn.Parameter(torch.FloatTensor([2.]))  # 2.

        else:
            self.lambda_cst = nn.Parameter(torch.FloatTensor([parameters['lambda_cst']]))
            self.lambda_apt = nn.Parameter(torch.FloatTensor([parameters['lambda_apt']]))
            self.inv_sigma = nn.Parameter(torch.FloatTensor([parameters['inv_sigma']]))
            self.alpha = nn.Parameter(torch.FloatTensor([parameters['alpha']]))

        self.conv_3x3 = nn.Conv2d(num_input_channel, 2, (3, 3), padding=1).type(torch.DoubleTensor)
        self.conv_3x3.weight.data[:, ] = 0.
        self.conv_3x3.weight.data[0, :, 1, 1] = -1.
        self.conv_3x3.weight.data[0, :, 1, 2] = 1.
        self.conv_3x3.weight.data[1, :, 1, 1] = -1.
        self.conv_3x3.weight.data[1, :, 2, 1] = 1.

    def parameters_constraint(self):
        """
        Apply box constraint on parameters
        :return
        """

        self.lambda_cst.data.clamp(0., 1000.)
        self.lambda_apt.data.clamp(0., 1000.)
        self.inv_sigma.data.clamp(0., 1000.)
        self.alpha.data.clamp(0.1, 10.)

    def forward(self, img, type=torch.DoubleTensor):
        """
        Compute the weights from image.
        :param img: Pytorch Variable [BxCxHxW]
        :return:Pytorch Variable
        """
        ttt = self.conv_3x3.forward(img)
        img_map = ttt ** 2.
        # img_map = torch.pow(ttt, self.alpha)
        img_map = img_map.permute(0, 2, 3, 1)
        w_adapt = torch.exp(- self.inv_sigma.type_as(img) * img_map)

        return self.lambda_cst.type_as(img) + self.lambda_apt.type_as(img) * \
                                              w_adapt


if __name__ == '__main__':
    raise NotImplementedError()
