
# Pytorch module
# Input: #label L
# Ouput: distance matrix LxL or 1xLxL
# model: d(a, a) = 0 and d(a, b) >= 0 [ and optional d(a, b) <= d(a, c) + d(c, b) ]
# use d(a. b) = \phi(a-b) with \phi(x) >= 0 and monotone.
# Start with \phi(x) = \min( \alpha |x| + \beta x^2, \gamma)


import torch
import torch.nn as nn
from torch.autograd import Variable


class Distance(nn.Module):
    """
    Compute the distance matrix for a given # of labels:
        d(a, b) = \alpha \min( |a-b|, \beta)
    """

    def __init__(self, parameters=None):
        super(Distance, self).__init__()

        if not parameters:
            self.alpha = nn.Parameter(torch.FloatTensor([1.]))
            self.beta = nn.Parameter(torch.FloatTensor([4.]))

        else:
            self.alpha = nn.Parameter(torch.FloatTensor([parameters['alpha']]))
            self.beta = nn.Parameter(torch.FloatTensor([parameters['beta']]))

    def paramterers_constraint(self):
        """
        Apply box constraint on parameters
        """

        self.alpha.data.clamp(0., 1000.)
        self.beta.data.clamp(1., 1000.)

    def phi(self, x):
        """
        Computes \alpha \min( |x|, \beta)
        """
        return self.alpha * torch.min(torch.abs(x), self.beta)

    def forward(self, x):
        return self.phi(x)

    def create_distance_matrix(self, L):
        """
        Compute the weights from image
        """
        indices = Variable(torch.arange(start=0, end=L, step=1))
        dist = Variable(torch.FloatTensor(1, L, L)).zero_()
        for l in range(L):
            dist[0, :, l] = self.phi(indices - float(l))

        return dist


if __name__ == '__main__':

    raise NotImplementedError()
