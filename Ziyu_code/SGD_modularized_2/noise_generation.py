# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program generates a measurement noise sequence and an observation noise sequence for a path

import numpy as np


def observation_noise_generator(d_Z, N, S):
    #  d_Z -- dimension of observation, N -- number of total time steps, S -- covariance matrix of observation noise
    #  returns a sequence of gaussian observation noise with mean 0 covariance S \
    #  as a 2d array V, each column V[:,i] is a noise vector
    V = np.zeros((d_Z, N))
    mu = np.zeros(d_Z)
    for i in range(N):

        V[:, i] = np.random.multivariate_normal(mu, S)

    return V


def system_noise_generator(d_X, N, R):
    #  d_X -- dimension of state, N -- number of total time steps, R -- covariance matrix of system noise
    #  returns a sequence of gaussian system noise with mean 0 covariance R \
    #  as a 2d array W, each column W[:,i] is a noise vector
    W = np.zeros((d_X, N))
    mu = np.zeros(d_X)
    for i in range(N):

        W[:, i] = np.random.multivariate_normal(mu, R)

    return W

