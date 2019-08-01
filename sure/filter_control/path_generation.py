# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program generates one or more (vectorized) trajectories with filtering and control

import numpy as np


def path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U):
    #  X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    #  B -- control coefficient matrix, G -- control gain, K -- Kalman gain, \
    #  N -- number of total time steps, W -- sequence of system noise, V -- sequence of observation noise, \
    #  d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control
    #  returns
    #  a sequence of states as a 3d array X, each element X[i,:,:] is a d_X-by-1 state vector
    #  a sequence of observations as a 3d array Z, each element Z[i,:,:] is a d_Z-by-1 observation vector
    #  a sequence of controls as a 3d array U, each element U[i,:,:] is a d_U-by-1 control vector
    #  a sequence of estimated states as a 3d array X_hat, each element X_hat[i,:,:] is a d_X-by-1 estimation vector

    X = np.zeros((N + 1, d_X, 1))
    X_hat = np.zeros((N + 1, d_X, 1))
    Z = np.zeros((N + 1, d_Z, 1))
    U = np.zeros((N + 1, d_U, 1))

    X[0, :, :] = X0  # initial state
    X_hat[0, :, :] = X0  # initial state estimate

    for n in range(N):
        U[n, :, :] = G @ X_hat[n, :, :]  # control
        X[n + 1, :, :] = A @ X[n, :, :] + B @ U[n, :, :] + W[n, :, :]  # state update
        Z[n + 1, :, :] = C @ X[n + 1, :, :] + V[n, :, :]               # observation
        grp1 = A @ X_hat[n, :, :] + B @ U[n, :, :]
        X_hat[n + 1, :, :] = grp1 + K @ (Z[n + 1, :, :] - C @ grp1)    # state estimate

    return X, Z, U, X_hat


def multi_paths_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U):
    #  generates multiple paths
    #  returns 4d arrays X, Z, U, X_hat, \
    #  e.g. each X[i,:,:,:] is a path, and each X[i,i,:,:] is a d_X-by-1 state vector

    m = len(W)
    X = np.zeros((m, N + 1, d_X, 1))
    X_hat = np.zeros((m, N + 1, d_X, 1))
    Z = np.zeros((m, N + 1, d_Z, 1))
    U = np.zeros((m, N + 1, d_U, 1))

    X[:, 0, :, :] = X0 # initial state
    X_hat[:, 0, :, :] = X0  # initial state estimate

    for n in range(N):
        U[:, n, :, :] = G @ X_hat[:, n, :, :]  # control
        X[:, n + 1, :, :] = A @ X[:, n, :, :] + B @ U[:, n, :, :] + W[:, n, :, :]  # state update
        Z[:, n + 1, :, :] = C @ X[:, n + 1, :, :] + V[:, n, :, :]               # observation
        grp1 = A @ X_hat[:, n, :, :] + B @ U[:, n, :, :]
        X_hat[:, n + 1, :, :] = grp1 + K @ (Z[:, n + 1, :, :] - C @ grp1)    # state estimate

    return X, Z, U, X_hat


# from initialization import *
# import noise_generation
# m = 3
# N = 4
# K = np.array([[0.5], [0.5]])
# G = np.array([[-1.0, -0.1]])
# W = noise_generation.vectorized_system_noise_generator(m, d_X, N, R)
# V = noise_generation.vectorized_observation_noise_generator(m, d_Z, N, S)
# X1, Z1, U1, X_hat1 = path_generator(X0, A, C, B, G, K, N, W[1], V[1], d_X, d_Z, d_U)
# print(X1)
# X, Z, U, X_hat = multi_paths_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
# print(X[1])
