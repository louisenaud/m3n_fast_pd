import unittest
from src.m3n import M3N
from src.margin import Margin
from src.energy import Energy
from src.scorer import Scorer
from src.unaries import Unary
from src.pairwise import Pairwise
from src.weights import Weights
from src.distance_matrix import Distance
import torch
from torch.autograd import Variable


class TestM3N(unittest.TestCase):
    def setUp(self):
        B, C, H, W = (2, 3, 512, 512)

        self.inputs_shape = (B, C, H, W)
        self.outputs_shape = B

        self.margin = Margin()

        scorer = Scorer(num_input_channel=C)
        unary = Unary(scorer)

        weights = Weights(num_input_channel=C)
        distance = Distance()
        pairwise = Pairwise(weights, distance)

        self.energy = Energy(unary, pairwise)

    def create_inputs(self):
        B, C, H, W = self.inputs_shape
        img0_patch = Variable(torch.FloatTensor(B, C, H, W).zero_())
        img1_patch = Variable(torch.FloatTensor(B, C, H, W).zero_())
        x_gt = Variable(torch.FloatTensor(B, H, W).zero_())
        x_min = Variable(torch.FloatTensor(B, H, W).zero_())

        return img0_patch, img1_patch, x_gt, x_min

    def create_outputs(self):
        B = self.outputs_shape
        return Variable(torch.FloatTensor(B).zero_())

    def test_constructor(self):
        module = M3N(self.energy, self.margin)
        self.assertTrue(isinstance(module, M3N))

    def test_forward(self):
        img0, img1, x_gt, x_min = self.create_inputs()

        module = M3N(self.energy, self.margin)
        nrg = module.forward(img0, img1, x_gt, x_min)

        self.assertEqual(nrg.size()[0], self.outputs_shape)

    def test_train(self):
        # Inputs and target
        img0, img1, x_gt, x_min = self.create_inputs()
        target = self.create_outputs()

        # Criterion
        criterion = torch.nn.MSELoss()

        # Model
        module = M3N(self.energy, self.margin)

        # Optimizer
        optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)

        for _ in range(2):
            # Forward + loss
            outputs = module.forward(img0, img1, x_gt, x_min)
            loss = criterion.forward(outputs, target)

            # Backward + update
            loss.backward()
            optimizer.step()

            # Constraints
            module.paramterers_constraint()


if __name__ == '__main__':
    unittest.main()
