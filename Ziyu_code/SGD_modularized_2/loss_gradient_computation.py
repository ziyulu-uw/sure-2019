# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program computes the loss F and the gradient of F w.r.t. K for a given path

import numpy as np


def compute_loss(X_l, X_hat_l, N):
    # X_l -- list of states from one path, X_hat_l -- list of state estimations from one path, N -- number of time steps
    # computes the mean-squared error: F = 1/2N *\sum_{n=1}^N (X_hat_n-X_n)^2
    # returns F

    F = 0
    assert (N == len(X_l[0])-1), "Number of intended time steps and number of states not equal. Something is wrong."
    for i in range(1, N+1):
        error = X_hat_l[:, i] - X_l[:, i]
        F += (error[0]**2 + error[1]**2)
    return F/(2*N)


def compute_gradient(A, C, N, K, X_l, Z_l, X_hat_l):
    # A -- state transition matrix, C -- observation matrix, N -- number of total time steps, \
    # K -- transpose of Kalman gain, X_l -- list of states from one path, Z_l -- list of observations from one path, \
    # X_hat_l -- list of state estimations from one path
    # computes dF/dK, where F = 1/2N *\sum_{n=1}^N (X_hat_n-X_n)^2
    # returns dF/dK
    d_X = len(X_l)
    K = np.array(K, ndmin=2).transpose()  # reshape to the right dimension
    Q = np.array([0, 0])
    P = 2*(X_hat_l[:, N] - X_l[:, N])  # start with P_N = 2*(\hat{X_N} - X_N)^T

    for i in range(N-1, -1, -1):  # move backward
        diag = np.array(Z_l[:, i+1], ndmin=2).transpose() - C @ A @ np.array(X_hat_l[:, i], ndmin=2).transpose()
        diag = np.array([1]*d_X) * diag[0][0]
        Q = Q + P @ np.diag(diag)
        P = 2*(X_hat_l[:, i] - X_l[:, i]) + P @ (np.identity(2) - K @ C) @ A

    return Q/(2*N)
