# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program performs gradient testing and gradient descent

import numpy as np
from initialization import *
import noise_generation
import convergence_study
import wrappers
import ultimate_wrappers
import stability_check
import performance_test


###### gradient testing ######
# np.random.seed(4)  # set random seed so the result can be reproduced
# W = noise_generation.system_noise_generator(d_X, N, R)
# V = noise_generation.observation_noise_generator(d_Z, N, S)
# K = np.array([[1.0], [1.0]])  # Kalman gain
# G = np.array([[1.0, 1.0]])  # control gain
# convergence_study.conv_study_K(X0, A, C, B, G, K, N, r, W, V, d_X, d_Z, d_U, delta_K=1e-3, which='B')
# convergence_study.conv_study_G(X0, A, C, B, G, K, N, r, W, V, d_X, d_Z, d_U, delta_G=1e-4, which='B')
###### gradient descent ######
K = np.array([[0.5], [0.5]])  # initial Kalman gain
G = np.array([[-1.0, -0.1]])  # initial control gain
result = ultimate_wrappers.wrapper(X0, A, C, B, G, K, N, S, R, d_X, d_Z, d_U, r, n=1000, act=True, L=[200, 400, 600], g=0.1,
                                                                            s_l=[1, 4], alpha=0.1, momentum=0, M=1, comp=None, which='RMSprop')
# result = wrappers.wrapper(X0, A, C, B, G, K, N, S, R, d_X, d_Z, d_U, r, n=500, L=[200, 400, 600], g=0.1, s_l=[1], which='Adam', alpha=0.1, momentum=0)
if result is None:
    print("Can't do optimization since the dynamics is unstable.")
else:
    K_avg, G_avg, F_avg = result[0], result[1], result[2]
    test_loss = performance_test.test(X0, A, C, B, G_avg, K_avg, N, R, S, r, d_X, d_Z, d_U, n=1000)
