"""
Project:    m3n_fast_pd
File:       mrf.py
Created by: louise
"""
# Pytorch module
# Input: x_gt, x_min BxHxW and img_0, img_1 BxCxHxW
# Ouput: loss B
# model: E(x_gt) - E_bar(x_min)

import torch
import torch.nn as nn
import numpy as np
from fast_pd.fast_pd import PyFastPd


class MRF(nn.Module):
    def __init__(self, unary, pairwise):
        """

        :param unary:
        :param pairwise:
        """
        super(MRF, self).__init__()

        self.unary = unary
        self.pairwise = pairwise

    def get_pairwise_costs(self, img0, x_min, x_max):
        """

        :param img0:
        :param x_min:
        :param x_max:
        :return:
        """
        L = x_max - x_min + 1

        # Compute pairwise term
        w = self.pairwise.weights.forward(img0)
        dist_mat = self.pairwise.distance.create_distance_matrix(L)

        w = w.data.cpu().numpy()
        dist_mat = dist_mat.data.cpu().numpy()

        return w, dist_mat

    def get_unary_cost(self, img0_patch, img1, x_disp, x_gt=None, margin=None):
        """

        :param img0_patch:
        :param img1:
        :param x_disp:
        :param x_gt:
        :param margin:
        :return:
        """
        # Extract patch form img1 for current x
        img1_patch = self.unary.patch_extractor.forward(img1, x_disp)

        # Compute unary cost between img0_patch and img1_patch
        unary_cost = self.unary.score.forward(img0_patch, img1_patch)

        del img1_patch

        # Add margin
        if x_gt is not None:
            margin_cost = margin.forward(x_gt, x_disp)
            unary_cost = unary_cost.unsqueeze(1) - margin_cost

        return unary_cost

    def get_unary_costs(self, img0, img1, x_min, x_max, x_gt=None, margin=None):
        """
        Computes unary costs for each image pair in the batch.
        :param img0: Pytorch Variable [BxCxHxW]
        :param img1: Pytorch Variable [BxCxHxW]
        :param x_min: Pytorch Variable [BxHxW]
        :param x_max: Pytorch Variable [BxHxW]
        :param x_gt: Pytorch Variable [BxHxW]
        :param margin: Margin
        :return: numpy array [BxHxW]
        """
        B, C, H, W = img0.size()

        L = x_max - x_min + 1
        unary_cost = torch.zeros(B, H, W, int(L))

        # Precompute img0_patch
        img0_patch = self.unary.patch_extractor.forward(img0)

        for l in range(int(L)):
            x_disp = x_min + l
            cost = self.get_unary_cost(img0_patch, img1, x_disp, x_gt, margin)
            unary_cost[:, :, :, l] = cost.data

        return unary_cost.cpu().numpy()

    def fast_pd_map(self, unary_cost, w, dist_mat):
        """
        Computes the MAP of a pairwise MRF.
        :param unary_cost: numpy array
        :param w: numpy array
        :param dist_mat: numpy array
        :return: numpy array
        """

        fast_pd_type = np.float32

        # Reshape inputs as vector
        R, C, L = unary_cost.shape
        unaries = unary_cost.reshape((R * C * L), order='F').astype(fast_pd_type)
        weights = w.reshape((R * C * 2), order='F').astype(fast_pd_type)
        dist = dist_mat.reshape((L * L), order='F').astype(fast_pd_type)
        x_init = np.zeros((R * C), order='F').astype(fast_pd_type)

        # Call fast PD solver
        fast_pd = PyFastPd(int(R), int(C), int(L), unaries, weights, dist, x_init)
        fast_pd.optimize(int(10), False)
        x_sol = fast_pd.get_solution()

        # Reshape solution
        x_sol = x_sol.reshape((R, C), order='F').astype(np.float32)

        return x_sol

    def map_inference(self, unary_cost, w, dist_mat):
        """
        Wrapper for C++ MAP inference FastPD.
        :param unary_cost: numpy array [BxHxWxL].
        :param w: numpy array [BxHxW].
        :param dist_mat: numpy array [BxHxW].
        :return: numpy array [BxHxW].
        """
        B, H, W, L = unary_cost.shape
        x_min = np.zeros((B, H, W))

        for b in range(B):
            x_sol = self.fast_pd_map(unary_cost[b, ], w[b, ], dist_mat[0, :, :])
            x_min[b, ] = x_sol

        return x_min

    def forward(self, img0, img1, x_min, x_max, x_opt=None, margin=None):
        """
        Computes the optimal label for each pixel.
        :param img0: Pytorch Variable [BxCxHxW]
        :param img1: Pytorch Variable [BxCxHxW]
        :param x_min: Pytorch Variable [1]
        :param x_max: Pytorch Variable [1]
        :param x_opt: numpy array [BxHxW]
        :param margin:
        :return: numpy array
        """
        # Pairwise costs
        w, dist_mat = self.get_pairwise_costs(img0, x_min, x_max)

        # Unary costs
        unary_cost = self.get_unary_costs(img0, img1, x_min, x_max, x_opt, margin)

        # Perform MAP inference
        x_sol = self.map_inference(unary_cost, w, dist_mat)

        return torch.from_numpy(x_sol + x_min)

    def backward(self):
        raise NotImplementedError('Can not backpropagate through MAP inference function (using C++ code)')
