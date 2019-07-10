# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program performs stochastic gradient descent on the Kalman filter optimization problem

import numpy as np
import noise_generation
import path_generation
import loss_gradient_computation


def stochastic_gradient_descent(X0, A, C, N, R, S, K, n, alpha, s):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # N -- number of total time steps, R -- system noise covariance matrix, S -- observation noise covariance matrix, \
    # K -- transpose of initial Kalman gain, n -- number of total gradient steps, \
    # alpha -- learning rate, s -- random seed
    # performs gradient descent using dF/dK as gradient
    # returns K^T, a list of F at each gradient step, a list of dF/dK at each gradient step

    np.random.seed(s)
    d_X = len(X0)  # dimension of state
    d_Z = len(C)  # dimension of observation
    F_l = []
    grad_l = []

    for i in range(n):

        W = noise_generation.system_noise_generator(d_X, N, R)
        V = noise_generation.observation_noise_generator(d_Z, N, S)
        X, Z = path_generation.path_generator(X0, A, C, N, W, V)
        X_hat = path_generation.filtered_path_generator(X0, A, C, K, Z, N)
        F = loss_gradient_computation.compute_loss(X, X_hat, N)
        grad = loss_gradient_computation.compute_gradient(A, C, N, K, X, Z, X_hat)
        F_l.append(F)
        grad_l.append(grad)
        # print("grad", grad)
        K = K - alpha * grad
        # print("K", K)

    return K, F_l, grad_l
