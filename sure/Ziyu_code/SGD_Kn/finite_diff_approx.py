# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program computes the gradient of F w.r.t. K using finite difference approximation

import numpy as np
import path_generation
import loss_gradient_computation


def finite_diff_approx(X0, A, C, N, W, V, K, n, delta_K):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, N -- number of total time steps, \
    # W -- sequence of system noise, V -- sequence of observation noise, K -- sequence of Kalman gains,\
    # n -- index of Kalman gain to take derivative with, delta_K -- difference in finite difference approximation
    # performs first order finite difference approximation of dF/dK_n: dF/dK_n = (F(K_n+delta_K) - F(K_n))/delta_K
    # returns the finite difference approximation of dF/dK_n
    Kn = K[:, n]
    grad_approx = np.array([0.0]*len(Kn))

    # compute F(K)
    X, Z = path_generation.path_generator(X0, A, C, N, W, V)
    X_hat = path_generation.filtered_path_generator(X0, A, C, K, Z, N)
    F = loss_gradient_computation.compute_loss(X, X_hat, N)

    # compute F(Kn+delta_K)
    for i in range(len(Kn)):

        K[:, n][i] += delta_K
        X_hat_ = path_generation.filtered_path_generator(X0, A, C, K, Z, N)
        F_ = loss_gradient_computation.compute_loss(X, X_hat_, N)
        grad_approx[i] = (F_ - F)/delta_K
        K[:, n][i] -= delta_K

    return grad_approx
