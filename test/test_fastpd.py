import unittest
from src.fast_pd.fast_pd import PyFastPd

import numpy as np

fast_pd_type = np.float32


class TestPyFastPd(unittest.TestCase):

    def setUp(self):
        R, C, L = (128, 20, 25)

        self.inputs_shape = (R, C, L)
        self.outputs_shape = (R, C)

    def create_inputs(self):

        R, C, L = self.inputs_shape
        R, C, L = int(R), int(C), int(L)

        unaries = 100 * np.random.rand(R, C, L).astype(fast_pd_type)
        weights = 100 * np.random.rand(R, C, 2).astype(fast_pd_type)

        dist = np.zeros((L, L), dtype=fast_pd_type)
        for a in range(L):
            for b in range(L):
                dist[a, b] = abs(a - b)

        x_init = np.zeros((R, C), dtype=fast_pd_type)

        # Temporary Reshape
        unaries = unaries.reshape((R * C * L), order='F')
        weights = weights.reshape((R * C * 2), order='F')
        dist = dist.reshape((L * L), order='F')
        x_init = x_init.reshape((R * C), order='F')

        return R, C, L, unaries, weights, dist, x_init

    def test_constructor(self):
        R, C, L, unaries, weights, dist, x_init = self.create_inputs()

        fast_pd = PyFastPd(R, C, L, unaries, weights, dist, x_init)
        self.assertTrue(isinstance(fast_pd, PyFastPd))

    def test_optimize(self):
        R, C, L, unaries, weights, dist, x_init = self.create_inputs()

        fast_pd = PyFastPd(R, C, L, unaries, weights, dist, x_init)
        fast_pd.optimize(int(10), True)
        fast_pd.restore_unaries()
        x_sol = fast_pd.get_solution()

        x_sol = x_sol.reshape((R, C), order='F')

        self.assertEqual(x_sol.shape, self.outputs_shape)


if __name__ == '__main__':
    unittest.main()
