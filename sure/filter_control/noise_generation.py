# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program generates measurement noise sequence and observation noise sequence for a path

import numpy as np


def system_noise_generator(d_X, N, R):
    #  d_X -- dimension of state, N -- number of total time steps, R -- covariance matrix of system noise
    #  returns a sequence of gaussian system noise with mean 0 covariance R \
    #  as a 3d array V, each element W[i,:,:] is a d_X-by-1 noise vector

    mu = np.zeros(d_X)
    W = np.random.multivariate_normal(mu, R, N)
    W = np.reshape(W, [N, d_X, 1])

    return W


def observation_noise_generator(d_Z, N, S):
    #  d_Z -- dimension of observation, N -- number of total time steps, S -- covariance matrix of observation noise
    #  returns a sequence of gaussian observation noise with mean 0 covariance S \
    #  as a 3d array V, each element V[i,:,:] is a d_Z-by-1 noise vector

    mu = np.zeros(d_Z)
    V = np.random.multivariate_normal(mu, S, N)
    V = np.reshape(V, [N, d_Z, 1])

    return V


def vectorized_system_noise_generator(n, d_X, N, R):
    #  n -- number of paths, d_X -- dimension of state, N -- number of total time steps, \
    #  R -- covariance matrix of system noise
    #  returns n sequences of gaussian system noise with mean 0 covariance R \
    #  as a 4d array V, each W[i,:,:,:] is a sequence, and each W[i,i,:,:] is a d_X-by-1 noise vector

    mu = np.zeros(d_X)
    W = np.random.multivariate_normal(mu, R, [n, N])
    W = np.reshape(W, [n, N, d_X, 1])

    return W


def vectorized_observation_noise_generator(n, d_Z, N, S):
    #  n -- number of paths, d_Z -- dimension of observation, N -- number of total time steps, \
    #  S -- covariance matrix of observation noise
    #  returns n sequences of gaussian observation noise with mean 0 covariance S \
    #  as a 4d array V, each V[i,:,:,:] is a sequence, and each V[i,i,:,:] is a d_Z-by-1 noise vector

    mu = np.zeros(d_Z)
    V = np.random.multivariate_normal(mu, S, [n, N])
    V = np.reshape(V, [n, N, d_Z, 1])

    return V


# from initialization import *
# m = 1
# N = 4
# np.random.seed(1)
# W = vectorized_system_noise_generator(m, d_X, N, R)
# np.random.seed(1)
# W1 = system_noise_generator(d_X, N, R)
