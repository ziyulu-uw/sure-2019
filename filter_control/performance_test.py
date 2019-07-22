# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program tests the performance of the optimal Kalman gain and control gain

import noise_generation
import path_generation
import loss_computation


def test(X0, A, C, B, G, K, N, R, S, r, d_X, d_Z, d_U):
    #  X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    #  B -- control coefficient matrix, G -- optimal control gain, K -- optimal Kalman gain, \
    #  N -- number of total time steps,R -- covariance matrix of system noise, \
    #  S -- covariance matrix of observation noise, r -- scaling factor, \
    #  d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control
    #  tests the performance of K and G on 10 random filtering and control problems
    #  returns the average error

    avg_F = 0
    for i in range(10):
        W = noise_generation.system_noise_generator(d_X, N, R)
        V = noise_generation.observation_noise_generator(d_Z, N, S)
        X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
        F = loss_computation.compute_loss(X, U, N, r)
        avg_F += F

    avg_F = avg_F/10
    print("Testing result:{:10.2e}".format(avg_F))

    return avg_F