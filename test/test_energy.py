import unittest
from src.energy import Energy
from src.scorer import Scorer
from src.unaries import Unary
from src.pairwise import Pairwise
from src.weights import Weights
from src.distance_matrix import Distance
import torch
from torch.autograd import Variable


class TestEnergy(unittest.TestCase):
    def setUp(self):
        B, C, H, W = (2, 3, 10, 20)

        self.inputs_shape = (B, C, H, W)
        self.outputs_shape = B

        scorer = Scorer(num_input_channel=C)
        self.unary = Unary(scorer)

        weights = Weights(num_input_channel=C)
        distance = Distance()
        self.pairwise = Pairwise(weights, distance)

    def create_inputs(self):
        B, C, H, W = self.inputs_shape
        img0_patch = Variable(torch.DoubleTensor(B, C, H, W).zero_())
        img1_patch = Variable(torch.DoubleTensor(B, C, H, W).zero_())
        x = Variable(torch.DoubleTensor(B, H, W).zero_())

        return img0_patch, img1_patch, x

    def create_outputs(self):
        B = self.outputs_shape
        return Variable(torch.DoubleTensor(B).zero_())

    def test_constructor(self):
        module = Energy(self.unary, self.pairwise)
        self.assertTrue(isinstance(module, Energy))

    def test_forward(self):
        img0, img1, x = self.create_inputs()

        module = Energy(self.unary, self.pairwise)
        nrg = module.forward(img0, img1, x)

        self.assertEqual(nrg.size()[0], self.outputs_shape)

    def test_train(self):
        # Inputs and target
        img0, img1, x = self.create_inputs()
        target = self.create_outputs()

        # Criterion
        criterion = torch.nn.MSELoss()

        # Model
        module = Energy(self.unary, self.pairwise)

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
            module.parameters_constraint()


if __name__ == '__main__':
    unittest.main()
