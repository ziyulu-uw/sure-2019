# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program performs gradient testing and gradient descent

import numpy as np
from initialization import *
import noise_generation
import convergence_study


###### gradient testing ######
np.random.seed(1)  # set random seed so the result can be reproduced
W = noise_generation.system_noise_generator(d_X, N, R)
V = noise_generation.observation_noise_generator(d_Z, N, S)
K = np.array([[1.0], [1.0]])  # Kalman gain
G = np.array([[1.0, 1.0]])  # control gain
# convergence_study.conv_study_K(X0, A, C, B, G, K, N, r, W, V, d_X, d_Z, d_U, delta_K=1e-2, which='B')
convergence_study.conv_study_G(X0, A, C, B, G, K, N, r, W, V, d_X, d_Z, d_U, delta_G=1e-2, which='B')
###### gradient descent ######
