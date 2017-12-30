
import unittest
from src.scorer import Scorer
from src.unaries import Unary
import torch
from torch.autograd import Variable


class TestUnary(unittest.TestCase):

    def setUp(self):

        B, C, H, W = (2, 3, 10, 20)

        self.inputs_shape = (B, C, H, W)
        self.outputs_shape = (B, H, W)
        self.scorer = Scorer(num_input_channel=C)

    def create_inputs(self):
        B, C, H, W = self.inputs_shape
        img0_patch = Variable(torch.FloatTensor(B, C, H, W).zero_())
        img1_patch = Variable(torch.FloatTensor(B, C, H, W).zero_())
        x = Variable(torch.FloatTensor(B, H, W).zero_())

        return img0_patch, img1_patch, x

    def create_outputs(self):
        B, H, W = self.outputs_shape
        return Variable(torch.FloatTensor(B, H, W).zero_())

    def test_constructor(self):

        module = Unary(self.scorer)
        self.assertTrue(isinstance(module, Unary))

    def test_forward(self):

        img0, img1, x = self.create_inputs()
        module = Unary(self.scorer)
        unary = module.forward(img0, img1, x)

        self.assertEqual(unary.size(), self.outputs_shape)

    def test_train(self):

        # Inputs and target
        img0, img1, x = self.create_inputs()
        target = self.create_outputs()

        # Criterion
        criterion = torch.nn.MSELoss()

        # Model
        module = Unary(self.scorer)

        # Optimizer
        optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)

        for _ in range(2):
            # Forward + loss
            outputs = module.forward(img0, img1, x)
            loss = criterion.forward(outputs, target)

            # Backward + update
            loss.backward()
            optimizer.step()

            # Constraints
            module.paramterers_constraint()

if __name__ == '__main__':
    unittest.main()


