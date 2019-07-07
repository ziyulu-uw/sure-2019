# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program generates a trajectory and a filtered trajectory

import numpy as np


def path_generator(X0, A, C, N, W, V):
    #  X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    #  N -- number of total time steps, W -- sequence of system noise, V -- sequence of observation noise
    #  returns a sequence of states as a 2d array X_l, each column X_l[:,i] is a state vector
    #  and a sequence of observations as a 2d array Z_l, each column Z_l[:,i] is an observation vector
    d_X = len(X0)  # dimension of state
    d_Z = len(C)  # dimension of observation
    X_l = np.zeros((d_X, N+1))
    Z_l = np.zeros((d_Z, N+1))
    X_l[:, 0] = X0  # initial state

    for n in range(N):

        X = np.array(X_l[:, n], ndmin=2).transpose()  # reshape to the right dimension
        X = A @ X + np.array(W[:, n], ndmin=2).transpose()  # state update
        X_l[:, n+1] = X.transpose()  # stores the new state in the state list
        Z = C @ X + np.array(V[:, n], ndmin=2).transpose()  # observation
        Z_l[:, n+1] = Z.transpose()  # stores the new observation in the observation list

    return X_l, Z_l


def filtered_path_generator(X0, A, C, K, Z_l, N):
    #  X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    #  K -- transpose of Kalman gain, Z_l -- sequence of observations, N -- number of total time steps
    #  returns a sequence of estimated states as a 2d array X_hat_l, each column X_hat_l[:,i] is a state vector
    d_X = len(X0)  # dimension of state
    X_hat_l = np.zeros((d_X, N+1))
    X_hat_l[:, 0] = X0  # initial state estimate
    K = np.array(K, ndmin=2).transpose()  # reshape to the right dimension

    for n in range(N):

        X_hat = np.array(X_hat_l[:, n], ndmin=2).transpose()  # reshape to the right dimension
        Z_hat = C @ A @ X_hat  # predicted observation
        Z = np.array(Z_l[:, n+1], ndmin=2).transpose()  # reshape to the right dimension
        X_hat = A @ X_hat + K @ (Z - Z_hat)  # state estimate
        X_hat_l[:, n+1] = X_hat.transpose()  # stores the new estimation in the estimation list

    return X_hat_l






