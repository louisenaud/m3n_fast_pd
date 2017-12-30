import unittest
from src.margin import Margin
import torch
from torch.autograd import Variable


class TestMargin(unittest.TestCase):
    def setUp(self):
        B, H, W = (2, 10, 20)

        self.inputs_shape = (B, H, W)
        self.outputs_shape = (B)

    def create_inputs(self):
        B, H, W = self.inputs_shape

        x_ref = Variable(torch.FloatTensor(B, H, W).zero_())
        x = Variable(torch.FloatTensor(B, H, W).zero_())

        return x_ref, x

    def create_outputs(self):
        B = self.outputs_shape
        return Variable(torch.FloatTensor(B).zero_())

    def test_constructor(self):
        module = Margin()
        self.assertTrue(isinstance(module, Margin))

    def test_forward(self):
        x_ref, x = self.create_inputs()
        module = Margin()
        margin_cost = module.forward(x_ref, x)

        self.assertEqual(margin_cost.size()[0], self.outputs_shape)

if __name__ == '__main__':
    unittest.main()


