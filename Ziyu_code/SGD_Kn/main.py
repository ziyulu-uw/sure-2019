# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program performs gradient testing and SGD

import numpy as np
from initialization import *
import noise_generation
import convergence_study
import SGD_wrapper
import filter_tester


###### gradient testing ######
d_X = len(X0)  # dimension of state
d_Z = len(C)  # dimension of observation
np.random.seed(1)  # set random seed so the result can be reproduced
W = noise_generation.system_noise_generator(d_X, N, R)
V = noise_generation.observation_noise_generator(d_Z, N, S)
K = np.zeros((d_X, N+1))  # sequence of Kalman gains
for i in range(1, N+1):
    K[:, i] = np.array([1.0, 1.0])
for i in range(1, N+1):
    convergence_study.conv_study(X0, A, C, N, dt, W, V, K, delta_K=1e-6, m=i)

###### SGD ######
K = np.zeros((d_X, N+1))  # sequence of Kalman gains
for i in range(1, N+1):
    K[:, i] = np.array([1.0, 1.0])
final_K, F_L = SGD_wrapper.Stochastic_gradient_descent(X0, A, C, N, R, S, K0=K, n=1000, alpha=0.1, s_l=[1, 5, 10, 20, 27])
test_loss = filter_tester.test(X0, A, C, N, R, S, final_K)
