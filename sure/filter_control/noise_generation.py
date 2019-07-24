# Author: Ziyu Lu, Xinyu Li
# Email: zl1546@nyu.edu, xl1796@nyu.edu
# Date: July 2019
# Description: This program generates a measurement noise sequence and an observation noise sequence for a path

import numpy as np


def observation_noise_generator(d_Z, N, S):
    #  d_Z -- dimension of observation, N -- number of total time steps, S -- covariance matrix of observation noise
    #  returns a sequence of gaussian observation noise with mean 0 covariance S \
    #  as a 3d array V, each element V[i,:,:] is a d_Z-by-1 noise vector

    mu = np.zeros(d_Z)
    V = np.random.multivariate_normal(mu, S, N)
    V = np.reshape(V, [N, d_Z, 1])

    return V


def system_noise_generator(d_X, N, R):
    #  d_X -- dimension of state, N -- number of total time steps, R -- covariance matrix of system noise
    #  returns a sequence of gaussian system noise with mean 0 covariance R \
    #  as a 3d array V, each element W[i,:,:] is a d_X-by-1 noise vector

    mu = np.zeros(d_X)
    W = np.random.multivariate_normal(mu, R, N)
    W = np.reshape(W, [N, d_X, 1])

    return W
