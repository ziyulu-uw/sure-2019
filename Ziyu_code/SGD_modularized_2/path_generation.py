# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program generates a trajectory and a filtered trajectory

import numpy as np


def path_generator(X0, A, C, N, W, V):
    #  X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    #  N -- number of total time steps, W -- sequence of system noise, V -- sequence of observation noise
    #  returns a sequence of states as a 2d array X, each column X[:,i] is a state vector
    #  and a sequence of observations as a 2d array Z, each column Z[:,i] is an observation vector
    d_X = len(X0)  # dimension of state
    d_Z = len(C)  # dimension of observation
    X = np.zeros((d_X, N+1))
    Z = np.zeros((d_Z, N+1))
    X[:, 0] = X0  # initial state

    for n in range(N):

        X[:, n+1] = A @ X[:, n] + W[:, n]  # state update
        Z[:, n+1] = C @ X[:, n+1] + V[:, n]  # observation

    return X, Z


def filtered_path_generator(X0, A, C, K, Z, N):
    #  X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    #  K -- transpose of Kalman gain, Z -- sequence of observations, N -- number of total time steps
    #  returns a sequence of estimated states as a 2d array X_hat, each column X_hat[:,i] is a state vector
    d_X = len(X0)  # dimension of state
    X_hat = np.zeros((d_X, N+1))
    X_hat[:, 0] = X0  # initial state estimate
    K = np.array(K, ndmin=2)  # add one dimension

    for n in range(N):

        Z_hat = C @ A @ X_hat[:, n]  # predicted observation
        X_hat[:, n+1] = A @ X_hat[:, n] + (Z[:, n+1] - Z_hat) @ K  # state estimate

    return X_hat

