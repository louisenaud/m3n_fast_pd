
import unittest
from src.pairwise import Pairwise
from src.weights import Weights
from src.distance_matrix import Distance
import torch
from torch.autograd import Variable


class TestPairwise(unittest.TestCase):

    def setUp(self):

        B, C, H, W = (2, 3, 10, 20)

        self.inputs_shape = (B, C, H, W)
        self.outputs_shape = (B, H, W, 2)
        self.weights = Weights(num_input_channel=C)
        self.distance = Distance()

    def create_inputs(self):
        B, C, H, W = self.inputs_shape
        img0_patch = Variable(torch.DoubleTensor(B, C, H, W).zero_())
        x = Variable(torch.DoubleTensor(B, H, W).zero_())

        return img0_patch, x

    def create_outputs(self):
        B, H, W, C = self.outputs_shape
        return Variable(torch.DoubleTensor(B, H, W, C).zero_())

    def test_constructor(self):

        module = Pairwise(self.weights, self.distance)
        self.assertTrue(isinstance(module, Pairwise))

    def test_forward(self):

        img0, x = self.create_inputs()
        module = Pairwise(self.weights, self.distance)
        cost = module.forward(img0, x)

        self.assertEqual(cost.size(), self.outputs_shape)

    def test_train(self):

        # Inputs and target
        img0, x = self.create_inputs()
        target = self.create_outputs()

        # Criterion
        criterion = torch.nn.MSELoss()

        # Model
        module = Pairwise(self.weights, self.distance)

        # Optimizer
        optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)

        for _ in range(2):
            # Forward + loss
            outputs = module.forward(img0, x)
            loss = criterion.forward(outputs, target)

            # Backward + update
            loss.backward()
            optimizer.step()

            # Constraints
            module.parameters_constraint()


if __name__ == '__main__':
    unittest.main()


