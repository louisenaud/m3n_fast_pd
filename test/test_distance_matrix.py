import unittest
from src.distance_matrix import Distance
import torch
from torch.autograd import Variable


class TestDistance(unittest.TestCase):
    def setUp(self):
        B, H, W = (3, 10, 20)
        L = 10

        self.L = L
        self.inputs_shape = (B, H, W)
        self.outputs_shape = (B, H, W)
        self.matrix_shape = (1, L, L)

    @staticmethod
    def parameters():
        parameters = {'alpha': 1.,
                      'beta': 4.,
                      }

        return parameters

    def create_inputs(self):
        B, H, W = self.inputs_shape
        return Variable(torch.FloatTensor(B, H, W).zero_())

    def create_outputs(self):
        B, H, W = self.outputs_shape
        return Variable(torch.FloatTensor(B, H, W).zero_())

    def test_constructor(self):
        module = Distance()
        self.assertTrue(isinstance(module, Distance))

        module = Distance(self.parameters())
        self.assertTrue(isinstance(module, Distance))

    def test_forward(self):
        x = self.create_inputs()

        module = Distance()
        dx = module.forward(x)

        self.assertEqual(dx.size(), self.outputs_shape)

    def test_create_distance_matrix(self):
        module = Distance()
        matrix = module.create_distance_matrix(self.L)

        self.assertEqual(matrix.size(), self.matrix_shape)

    def test_train(self):
        # Inputs and target
        x = self.create_inputs()
        target = self.create_outputs()

        # Criterion
        criterion = torch.nn.MSELoss()

        # Model
        module = Distance()

        # Optimizer
        optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)

        for _ in range(2):
            # Forward + loss
            outputs = module.forward(x)
            loss = criterion.forward(outputs, target)

            # Backward + update
            loss.backward()
            optimizer.step()

            # Constraints
            module.parameters_constraint()


if __name__ == '__main__':
    unittest.main()
