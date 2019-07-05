# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program generates a measurement noise sequence and an observation noise sequence for a path

import numpy as np


def observation_noise_generator(d_Z, N, S):
    #  d_Z -- dimension of observation, N -- number of total time steps, S -- covariance matrix of observation noise
    #  returns a sequence of gaussian observation noise with mean 0 covariance S
    V_l = []
    for i in range(N):

        V_n = np.random.multivariate_normal([0]*d_Z, S)
        V_n = np.array(V_n, ndmin=2)
        V_n = np.transpose(V_n)
        V_l.append(V_n)

    return V_l


def system_noise_generator(d_X, N, R):
    #  d_X -- dimension of state, N -- number of total time steps, R -- covariance matrix of system noise
    #  returns a sequence of gaussian system noise with mean 0 covariance R
    W_l = []
    for i in range(N):

        W_n = np.random.multivariate_normal([0]*d_X, R)
        W_n = np.array(W_n, ndmin=2)
        W_n = np.transpose(W_n)
        W_l.append(W_n)

    return W_l

