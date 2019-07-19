# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program computes the loss F and the gradient of F w.r.t. K for a given path

import numpy as np


def compute_loss(X, U, N, r):
    # X -- list of states from one path, U -- list of controls from one path, N -- number of time steps
    # computes the mean-squared error: F = 1/2N *(X_N^TX_N + \sum_{n=0}^{N-1} (X_n^TX_n + rU_n^TU_n))
    # returns F

    F = 0
    assert (N == len(X[0])-1), "Number of intended time steps and number of states not equal. Something is wrong."
    for i in range(N+1):
        error = np.dot(X[:, i], X[:, i]) + r*U[:, i]**2
        F += (error[0]**2 + error[1]**2)

    return F/(2*N)


def compute_gradient(A, C, N, K, X, Z, X_hat):
    # A -- state transition matrix, C -- observation matrix, N -- number of total time steps, \
    # K -- transpose of Kalman gain, X -- list of states from one path, Z -- list of observations from one path, \
    # X_hat -- list of state estimations from one path
    # computes dF/dK, where F = 1/2N *\sum_{n=1}^N (X_hat_n-X_n)^2
    # returns dF/dK
    d_X = len(X)
    K = np.array(K, ndmin=2).transpose()  # reshape to the right dimension
    Q = 0
    P = 2*(X_hat[:, N] - X[:, N])  # start with P_N = 2*(\hat{X_N} - X_N)^T

    for i in range(N-1, -1, -1):  # move backward
        diag = Z[:, i+1] - C @ A @ X_hat[:, i]
        diag = np.array([1]*d_X) * diag
        Q = Q + P @ np.diag(diag)
        P = 2*(X_hat[:, i] - X[:, i]) + P @ (np.identity(2) - K @ C) @ A

    return Q/(2*N)
