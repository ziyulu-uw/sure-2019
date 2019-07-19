# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program tests if the forward and backward gradient computations give the same result

import numpy as np
from initialization import *
import noise_generation as noise
import path_generation
import backward_grad
import forward_grad

grad_diff_1 = []
grad_diff_2 = []
err_diff = []

for i in range(1):

    W = noise.system_noise_generator(d_X, N, R)
    V = noise.observation_noise_generator(d_Z, N, S)
    K = np.array([[1.0], [1.0]])
    G = np.array([[1.0, 1.0]])
    X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
    B_grad_K, B_grad_G = backward_grad.compute_gradient(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X)
    F_grad_K = forward_grad.filter_forward(X, X_hat, Z, A, B, C, G, N, K, r, d_X)
    F_grad_G = forward_grad.control_forward(X, X_hat, A, B, C, G, N, K, r, d_X)

    print(B_grad_K, B_grad_G)
    print(F_grad_K, F_grad_G)

    # K11 = 1 + np.random.rand()
    # K12 = 1 + np.random.rand()
    # seed = np.random.randint(2000)
    # K = np.array([[K11], [K12]])
    #
    # gradF, errF = forward.compute_gradient(K, seed)
    #
    # L_state, L_obs, L_est = backward.generate_path(K, seed)
    # gradB = backward.compute_gradient(K, L_state, L_obs, L_est)
    # errB = backward.compute_error(L_state, L_est)

    # print(gradF - gradB)
    # print(abs(errF - errB))

    # grad_diff = gradF - gradB
    #
    # grad_diff_1.append(abs(grad_diff[0][0]))
    # grad_diff_2.append(abs(grad_diff[0][1]))
    #
    # err_diff.append(abs(errF - errB))

# print(max(grad_diff_1))
# print(max(grad_diff_2))
# print(np.mean(grad_diff_1))
# print(np.mean(grad_diff_2))
# print(np.mean(err_diff))