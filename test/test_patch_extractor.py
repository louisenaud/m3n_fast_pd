
import unittest
from src.patch_extractor import PatchExtractor
import torch
from torch.autograd import Variable


class TestPatchExtractor(unittest.TestCase):

    def setUp(self):

        B, C, H, W = (2, 3, 10, 20)
        Hp, Wp = (3, 5)

        self.patch_shape = (Hp, Wp)
        self.inputs_shape = (B, C, H, W)
        self.outputs_shape = (B, C, H, W, Hp, Wp)

    def create_outputs(self):
        B, C, H, W, Hp, Wp = self.outputs_shape

        img = Variable(torch.FloatTensor(B, C, H, W, Hp, Wp).zero_())

        return img

    def create_inputs(self):
        B, C, H, W = self.inputs_shape

        img = Variable(torch.FloatTensor(B, C, H, W).zero_())
        x = Variable(1.0 + torch.FloatTensor(B, H, W).zero_())

        return img, x

    def test_constructor(self):

        module = PatchExtractor()
        self.assertTrue(isinstance(module, PatchExtractor))

        module = PatchExtractor(self.patch_shape)
        self.assertTrue(isinstance(module, PatchExtractor))

    def test_forward(self):

        img, x = self.create_inputs()
        module = PatchExtractor(self.patch_shape)
        img_patch = module.forward(img, x)

        self.assertEqual(img_patch.size(), self.outputs_shape)

        img_patch = module.forward(img)
        self.assertEqual(img_patch.size(), self.outputs_shape)


if __name__ == '__main__':
    unittest.main()


