# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program tests the performance of the optimal Kalman gain found by SGD

import noise_generation
import path_generation
import loss_gradient_computation


def test(X0, A, C, N, R, S, K):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # N -- number of total time steps, R -- system noise covariance matrix, S -- observation noise covariance matrix, \
    # K -- transpose of optimal Kalman gain
    # tests the performance of K on 10 random Kalman filtering problems
    # returns the average error
    # this error should be approximately equal to the final loss of SGD

    d_X = len(X0)  # dimension of state
    d_Z = len(C)  # dimension of observation
    avg_F = 0
    for i in range(10):
        W = noise_generation.system_noise_generator(d_X, N, R)
        V = noise_generation.observation_noise_generator(d_Z, N, S)
        X, Z = path_generation.path_generator(X0, A, C, N, W, V)
        X_hat = path_generation.filtered_path_generator(X0, A, C, K, Z, N)
        F = loss_gradient_computation.compute_loss(X, X_hat, N)
        avg_F += F

    avg_F = avg_F/10
    print("Testing result:{:10.2e}".format(avg_F))

    return avg_F
