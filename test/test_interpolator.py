import unittest
from src.interpolator import Interpolator
import torch
from torch.autograd import Variable


class TestInterpolator(unittest.TestCase):
    def setUp(self):
        B, C, H, W = (2, 3, 10, 20)

        self.inputs_shape = (B, C, H, W)
        self.outputs_shape = (B, C, H, W)

    def create_outputs(self):
        B, C, H, W = self.outputs_shape

        img = Variable(torch.FloatTensor(B, C, H, W).zero_())

        return img

    def create_inputs(self):
        B, C, H, W = self.inputs_shape

        img = Variable(torch.FloatTensor(B, C, H, W).zero_())
        x = Variable(1.0 + torch.FloatTensor(B, H, W, 2).zero_())

        return img, x

    def test_constructor(self):
        module = Interpolator()
        self.assertTrue(isinstance(module, Interpolator))

    def test_make_grid(self):
        B, C, H, W = self.inputs_shape

        module = Interpolator()
        grid = module.make_grid(H, W)
        grid = grid.data.numpy()

        self.assertEqual(grid[0, 0, 1, 0], 1)
        self.assertEqual(grid[0, 1, 0, 1], 1)

    def test_make_pytorch_grid(self):
        B, C, H, W = self.inputs_shape

        module = Interpolator()
        grid = module.make_pytorch_grid(H, W)
        grid = grid.data.numpy()

        self.assertAlmostEqual(grid[0, 0, 0, 0], -1.)
        self.assertAlmostEqual(grid[0, 0, 0, 1], -1.)

        self.assertAlmostEqual(grid[0, 0, -1, 0], 1.)
        self.assertAlmostEqual(grid[0, -1, 0, 1], 1.)

    def test_interp_grid(self):
        B, C, H, W = self.inputs_shape

        img, x = self.create_inputs()

        module = Interpolator()
        interp_grid = module.interp_grid(x)
        interp_grid = interp_grid.data.numpy()

        self.assertAlmostEqual(interp_grid[0, 0, 0, 1], -1. + 2. / float(H - 1))
        self.assertAlmostEqual(interp_grid[0, 0, 0, 0], -1. + 2. / float(W - 1))

    def test_forward(self):
        img, x = self.create_inputs()
        module = Interpolator()
        img_interp = module.forward(img, x)


if __name__ == '__main__':
    unittest.main()
