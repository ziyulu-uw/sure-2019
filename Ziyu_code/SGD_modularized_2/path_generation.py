# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program generates a trajectory and a filtered trajectory

import numpy as np


def path_generator(X0, A, C, N, W_l, V_l):
    #  X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    #  N -- number of total time steps, W_l -- sequence of system noise, V_l -- sequence of observation noise
    #  returns a sequence of states and a sequence of observations
    d_X = len(X0)  # dimension of state
    d_Z = len(C)  # dimension of observation
    X_l = []
    Z_l = []
    X = X0  # initial state
    X_l.append(X)
    Z_l.append(np.array([0]*d_Z, ndmin=2).transpose())
    for n in range(N):

        X = A @ X + W_l[n]  # state update
        X_l.append(X)  # stores the new state in the state list
        Z = C @ X + V_l[n]  # observation
        Z_l.append(Z)  # stores the new observation in the observation list

    return X_l, Z_l


def filtered_path_generator(X0, A, C, K, Z_l, N):
    #  X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    #  K -- Kalman gain, Z_l -- sequence of observations, N -- number of total time steps
    #  returns a sequence of estimated states
    X_hat = X0  # initial state estimate
    X_hat_l = []
    X_hat_l.append(X_hat)
    for n in range(N):

        Z_hat = C @ A @ X_hat  # predicted observation
        X_hat = A @ X_hat + K @ (Z_l[n+1] - Z_hat)  # state estimate
        X_hat_l.append(X_hat)  # stores the new estimation in the estimation list

    return X_hat_l






