# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program generates a trajectory and a filtered trajectory

import numpy as np
# from initialization import *
# import noise_generation as noise


def path_generator(X0, A, C, B, d_U, G, K, N, W, V):
    #  X0 -- initial state, A -- state transition matrix, C -- observation matrix, B -- control coefficient matrix, \
    #  d_U -- dimension of control, G -- control matrix, K -- transpose of Kalman gain, \
    #  N -- number of total time steps, W -- sequence of system noise, V -- sequence of observation noise
    #  returns a sequence of states as a 2d array X, each column X[:,i] is a state vector
    #  a sequence of observations as a 2d array Z, each column Z[:,i] is an observation vector
    #  a sequence of controls as a 2d array U, each column U[:,i] is a control vector
    #  a sequence of estimated states as a 2d array X_hat, each column X_hat[:,i] is a state estimation vector

    d_X = len(X0)  # dimension of state
    d_Z = len(C)  # dimension of observation
    X = np.zeros((d_X, N+1))
    X_hat = np.zeros((d_X, N+1))
    Z = np.zeros((d_Z, N+1))
    U = np.zeros((d_U, N))

    X[:, 0] = X0  # initial state
    X_hat[:, 0] = X0  # initial state estimate

    K = np.array(K, ndmin=2)  # add one dimension
    G_ = np.array(K, ndmin=2)  # add one dimension
    B_ = np.array(B, ndmin=2).transpose()  # reshape

    for n in range(N):

        U[:, n] = G@X[:, n]  # control
        X[:, n+1] = A @ X[:, n] + B*U[:, n] + W[:, n]  # state update
        Z[:, n+1] = C @ X[:, n+1] + V[:, n]  # observation
        coef = A + B_@G_
        Z_hat = C @ coef @ X_hat[:, n]  # predicted observation
        X_hat[:, n+1] = coef @ X_hat[:, n] + (Z[:, n+1] - Z_hat) @ K  # state estimate

    return X, Z, U, X_hat

# W = noise.system_noise_generator(2, N, R)
# V = noise.observation_noise_generator(1, N, S)
# K = np.array([1.0, 1.0])
# G = np.array([1.0, 1.0])
# X, Z, U, X_hat = path_generator(X0, A, C, B, G, K, N, W, V)
# print(X)
# print(Z)
# print(U)
# print(X_hat)


