# Author: Ziyu Lu, Xinyu Li
# Email: zl1546@nyu.edu, xl1796@nyu.edu
# Date: July 2019
# Description: This program generates a trajectory with filtering and control

import numpy as np
# from initialization import *
import noise_generation as noise


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

    X = np.zeros((N+1, d_X, 1))
    X_hat = np.zeros((N+1, d_X, 1))
    Z = np.zeros((N+1, d_Z, 1))
    U = np.zeros((N, d_U, 1))

    X[0,:,:] = X0  # initial state
    X_hat[0,:,:] = X0  # initial state estimate

    for n in range(N):

        U[n,:,:] = G @ X_hat[n,:,:]  # control
        X[n+1,:,:] = A @ X[n,:,:] + B @ U[n,:,:] + W[n,:,:]  # state update
        Z[n+1,:,:] = C @ X[n+1,:,:] + V[n,:,:]  # observation
        grp1 = A @ X_hat[n,:,:] + B @ U[n,:,:]
        X_hat[n+1,:,:] = grp1 + K @ (Z[n+1,:,:] - C @ grp1)  # state estimate

    return X, Z, U, X_hat

# W = noise.system_noise_generator(2, N, R)
# V = noise.observation_noise_generator(1, N, S)
# K = np.array([[1.0], [1.0]])
# G = np.array([[1.0, 1.0]])
# X, Z, U, X_hat = path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
# print('X:',X)
# print('Z:',Z)
# print('U:',U)
# print('X_hat:',X_hat)
