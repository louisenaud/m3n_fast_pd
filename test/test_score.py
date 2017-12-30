
import unittest
from src.scorer import Scorer
import torch
from torch.autograd import Variable


class TestScorer(unittest.TestCase):

    def setUp(self):

        B, C, H, W, Hp, Wp = (2, 3, 10, 20, 3, 5)

        self.inputs_shape = (B, C, H, W, Hp, Wp)
        self.outputs_shape = (B, H, W)

    @staticmethod
    def parameters():

        parameters = {'alpha': [1.2, 3]
                      }

        return parameters

    def create_inputs(self):
        B, C, H, W, Hp, Wp = self.inputs_shape
        img0_patch = Variable(torch.FloatTensor(B, C, H, W, Hp, Wp).zero_())
        img1_patch = Variable(torch.FloatTensor(B, C, H, W, Hp, Wp).zero_())

        return img0_patch, img1_patch

    def create_outputs(self):
        B, H, W = self.outputs_shape
        return Variable(torch.FloatTensor(B, H, W).zero_())

    def test_constructor(self):

        module = Scorer(num_input_channel=self.inputs_shape[1])
        self.assertTrue(isinstance(module, Scorer))

        module = Scorer(self.parameters())
        self.assertTrue(isinstance(module, Scorer))


    def test_forward(self):

        img0_patch, img1_patch = self.create_inputs()
        module = Scorer(num_input_channel=img0_patch.size()[1])
        score = module.forward(img0_patch, img1_patch)

        B_s, H_s, W_s = score.size()

        self.assertEqual(B_s, self.outputs_shape[0])
        self.assertEqual(H_s, self.outputs_shape[1])
        self.assertEqual(W_s, self.outputs_shape[2])

    def test_train(self):

        # Inputs and target
        img0_patch, img1_patch = self.create_inputs()
        target = self.create_outputs()

        # Criterion
        criterion = torch.nn.MSELoss()

        # Model
        module = Scorer(num_input_channel=img0_patch.size()[1])

        # Optimizer
        optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)

        for _ in range(2):
            # Forward + loss
            outputs = module.forward(img0_patch, img1_patch)
            loss = criterion.forward(outputs, target)

            # Backward + update
            loss.backward()
            optimizer.step()

            # Constraints
            module.paramterers_constraint()



if __name__ == '__main__':
    unittest.main()


