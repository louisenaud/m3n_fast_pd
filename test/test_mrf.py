import unittest
from src.mrf import MRF
from src.energy import Energy
from src.scorer import Scorer
from src.unaries import Unary
from src.pairwise import Pairwise
from src.weights import Weights
from src.distance_matrix import Distance
import torch
from torch.autograd import Variable


class TestMRF(unittest.TestCase):
    def setUp(self):
        B, C, H, W = (2, 3, 10, 20)

        self.inputs_shape = (B, C, H, W)
        self.outputs_shape = (B, H, W)

        scorer = Scorer(num_input_channel=C)
        self.unary = Unary(scorer)

        weights = Weights(num_input_channel=C)
        distance = Distance()
        self.pairwise = Pairwise(weights, distance)

    def create_inputs(self):
        B, C, H, W = self.inputs_shape
        img0 = Variable(torch.FloatTensor(B, C, H, W).zero_())
        img1 = Variable(torch.FloatTensor(B, C, H, W).zero_())
        x_min = 0
        x_max = 10

        return img0, img1, x_min, x_max

    def create_outputs(self):
        (B, H, W) = self.outputs_shape
        return Variable(torch.FloatTensor(B, H, W).zero_())

    def test_constructor(self):
        module = MRF(self.unary, self.pairwise)
        self.assertTrue(isinstance(module, MRF))

    def test_forward(self):
        img0, img1, x_min, x_max = self.create_inputs()

        module = MRF(self.unary, self.pairwise)
        x_min = module.forward(img0, img1, x_min, x_max)

        self.assertEqual(x_min.shape, self.outputs_shape)


if __name__ == '__main__':
    unittest.main()
