import unittest
from src.weights import Weights
import torch
from torch.autograd import Variable


class TestWeights(unittest.TestCase):
    def setUp(self):
        B, C, H, W = (2, 3, 10, 20)

        self.inputs_shape = (B, C, H, W)
        self.outputs_shape = (B, H, W, 2)

    @staticmethod
    def parameters():
        parameters = {'lambda_cst': 1.,
                      'lambda_apt': 10.,
                      'inv_sigma': 0.1,
                      'alpha': 1.2
                      }

        return parameters

    def create_inputs(self):
        B, C, H, W = self.inputs_shape
        return Variable(torch.DoubleTensor(B, C, H, W).zero_())

    def create_outputs(self):
        B, H, W, C = self.outputs_shape
        return Variable(torch.DoubleTensor(B, H, W, C).zero_())

    def test_constructor(self):
        weights = Weights()
        self.assertTrue(isinstance(weights, Weights))

        weights = Weights(self.parameters())
        self.assertTrue(isinstance(weights, Weights))

    def test_forward(self):
        img = self.create_inputs()
        weights = Weights(num_input_channel=img.size()[1])
        w = weights.forward(img)

        self.assertEqual(w.size(), self.outputs_shape)

    def test_train(self):
        # Inputs and target
        img = self.create_inputs()
        target = self.create_outputs()

        # Criterion
        criterion = torch.nn.MSELoss()

        # Model
        weights = Weights(num_input_channel=img.size()[1])

        # Optimizer
        optimizer = torch.optim.SGD(weights.parameters(), lr=1e-3)

        for _ in range(2):
            # Forward + loss
            outputs = weights.forward(img)
            loss = criterion.forward(outputs, target)

            # Backward + update
            loss.backward()
            optimizer.step()

            # Constraints
            weights.parameters_constraint()


if __name__ == '__main__':
    unittest.main()
