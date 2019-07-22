# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program tests if the forward and backward gradient computations give the same result
# by testing the computations on 1000 random problems

import numpy as np
from numpy import linalg as LA
from initialization import *
import noise_generation as noise
import path_generation
import backward_grad
import forward_grad

grad_K_diff_norm = []
grad_G_diff_norm = []

np.random.seed(2000)

for i in range(1000):

    W = noise.system_noise_generator(d_X, N, R)
    V = noise.observation_noise_generator(d_Z, N, S)
    K = np.random.rand(2, 1)
    G = np.random.rand(1, 2)
    X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
    B_grad_K, B_grad_G = backward_grad.compute_gradient(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X)
    F_grad_K = forward_grad.forward_K(X, X_hat, Z, U, A, B, C, G, K, N, d_X, r)
    F_grad_G = forward_grad.forward_G(X, X_hat, U, A, B, C, G, K, N, d_X, r)

    grad_K_diff = B_grad_K - F_grad_K
    # print(grad_K_diff)
    grad_G_diff = B_grad_G - F_grad_G
    # print(grad_G_diff)
    grad_K_diff_norm.append(LA.norm(grad_K_diff))
    grad_G_diff_norm.append(LA.norm(grad_G_diff))

print("Maximum difference norm: K: {:14.6e}  G:{:14.6e}".format(max(grad_K_diff_norm), max(grad_G_diff_norm)))
print("Average difference norm: K: {:14.6e}  G:{:14.6e}".format(np.mean(grad_K_diff_norm), np.mean(grad_G_diff_norm)))
