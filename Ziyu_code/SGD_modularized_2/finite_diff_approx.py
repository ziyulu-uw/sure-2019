# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program computes the gradient of F w.r.t. K using finite difference approximation

import numpy as np
import path_generation
import loss_gradient_computation


def finite_diff_approx(X0, A, C, N, W_l, V_l, K, delta_K):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, N -- number of total time steps, \
    # W_l -- sequence of system noise, V_l -- sequence of observation noise, \
    # K -- Kalman gain, delta_K -- difference in finite difference approximation
    # performs first order finite difference approximation of dF/dK: dF/dK = (F(K+delta_K) - F(K))/delta_K
    # returns the finite difference approximation of dF/dK

    grad_approx = np.array([[0.0]*len(K)])

    # compute F(K)
    X_l, Z_l = path_generation.path_generator(X0, A, C, N, W_l, V_l)
    X_hat_l = path_generation.filtered_path_generator(X0, A, C, K, Z_l, N)
    F = loss_gradient_computation.compute_loss(X_l, X_hat_l, N)

    # compute F(K+delta_K)
    for i in range(len(K)):

        K[i][0] += delta_K
        X_hat_l_ = path_generation.filtered_path_generator(X0, A, C, K, Z_l, N)
        F_ = loss_gradient_computation.compute_loss(X_l, X_hat_l_, N)
        grad_approx[0][i] = (F_ - F)/delta_K
        K[i][0] -= delta_K

    return grad_approx
