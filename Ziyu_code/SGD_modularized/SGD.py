# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program performs stochastic gradient descent on the Kalman filter optimization problem

import path_generation as pgen
import gradient_error_computation as comp
import numpy as np


def stochastic_gradient_descent(K, n, alpha, z):
    # K -- initial Kalman gain, n -- number of gradient steps, \
    # alpha -- learning rate, z -- random seed
    # performs gradient descent using dF/dK as gradient
    # returns K, a list of F at each gradient step, a list of dF/dK at each gradient step

    np.random.seed(z)  # set random seed
    err_L = []
    grad_L = []
    for i in range(n):
        L_state, L_obs, L_est = pgen.generate_path(K, z)
        grad = comp.compute_gradient(K, L_state, L_obs, L_est)
        err = comp.compute_error(L_state, L_est)
        err_L.append(err)
        grad_L.append(grad)
        # print(grad)
        K = K - alpha * np.transpose(grad)
        # print(K)

    return K, err_L, grad_L
