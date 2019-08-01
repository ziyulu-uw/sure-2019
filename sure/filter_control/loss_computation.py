# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program computes the loss F

import numpy as np
from backward_grad import transpose


def compute_loss(X, U, N, r):
    # X -- list of states from one path (X0, X1, ..., XN), 
    # U -- list of controls from one path (U0,U1,...,UN=0), 
    # Caution: U_N should be set as zero manually before call the function
    # N -- number of total time steps, r -- scaling factor
    # computes the mean-squared error: F = 1/2N \sum_{n=0}^{N} (X_n^TX_n + rU_n^TU_n), U_N=0
    # returns F

    F = 0
    assert (N == len(X) - 1), "Number of intended time steps and number of states not equal. Something is wrong."

    for i in range(N + 1):
        x = X[i, :, :]
        u = U[i, :, :]
        err1 = x.transpose() @ x
        err2 = u.transpose() @ u
        F += err1[0][0] + r*err2[0][0]

    return F / (2 * N)


def compute_multi_loss(X, U, N, r):
    # compute losses for multiple paths
    # returns a 3d array F, F[i,0,0] is the loss for the ith path

    m = len(X)
    F = np.zeros((m, 1, 1))
    assert (N == len(X[0]) - 1), "Number of intended time steps and number of states not equal. Something is wrong."

    for i in range(N + 1):
        x = X[:, i, :, :]
        u = U[:, i, :, :]
        err1 = transpose(x) @ x
        err2 = transpose(u) @ u
        F += err1 + r * err2

    return F / (2 * N)

# from initialization import *
# import noise_generation
# import path_generation
# m = 3
# N = 4
# K = np.array([[0.5], [0.5]])
# G = np.array([[-1.0, -0.1]])
# W = noise_generation.vectorized_system_noise_generator(m, d_X, N, R)
# V = noise_generation.vectorized_observation_noise_generator(m, d_Z, N, S)
# X1, Z1, U1, X_hat1 = path_generation.path_generator(X0, A, C, B, G, K, N, W[1], V[1], d_X, d_Z, d_U)
# F1 = compute_loss(X1, U1, N, r)
# print(F1)
# X, Z, U, X_hat = path_generation.multi_paths_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
# F = compute_multi_loss(X, U, N, r)
# print(F)