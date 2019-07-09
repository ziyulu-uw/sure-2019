# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program computes the loss F and the gradient of F w.r.t. K for a given path

import numpy as np


def compute_loss(X, X_hat, N):
    # X -- list of states from one path, X_hat -- list of state estimations from one path, N -- number of time steps
    # computes the mean-squared error: F = 1/2N *\sum_{n=1}^N (X_hat_n-X_n)^2
    # returns F

    F = 0
    assert (N == len(X[0])-1), "Number of intended time steps and number of states not equal. Something is wrong."
    for i in range(1, N+1):
        error = X_hat[:, i] - X[:, i]
        F += (error[0]**2 + error[1]**2)
    return F/(2*N)


def compute_gradient(A, C, N, K, X, Z, X_hat):
    # A -- state transition matrix, C -- observation matrix, N -- number of total time steps, \
    # K -- sequence of Kalman gains, X -- list of states from one path, Z -- list of observations from one path, \
    # X_hat -- list of state estimations from one path
    # computes dF/dK_n, where F = 1/2N *\sum_{n=1}^N (X_hat_n-X_n)^2
    # returns dF/dK_n for n=1,2,...,N, as a 2d array Q, each column Q[:,n] is dF/dK_n
    d_X = len(X)
    Q = np.zeros((d_X, N+1))  # a sequence of Q^m, for m=1,2,...,N
    P = np.zeros((d_X, N+1))  # a sequence of P_i, for i=1,2,...,N
    P[:, N] = 2*(X_hat[:, N] - X[:, N])  # start with P_N = 2*(\hat{X_N} - X_N)^T

    for i in range(N-1, 0, -1):  # move backward
        Kn = np.array(K[:, i+1], ndmin=2).transpose()  # reshape to the right dimension
        P[:, i] = 2*(X_hat[:, i] - X[:, i]) + P[:, i+1] @ (np.identity(2) - Kn @ C) @ A

    for m in range(N, 0, -1):
        diag = Z[:, m] - C @ A @ X_hat[:, m-1]
        diag = np.array([1] * d_X) * diag
        Q[:, m] = P[:, m] @ np.diag(diag)

    return Q/(2*N)
