# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program computes the gradients of F w.r.t. K and G for a given path using forward propagation

import numpy as np


def forward_K(X, X_hat, Z, U, A, B, C, G, K, N, d_X, r):
    # X -- list of states from one path, X_hat -- list of state estimations from one path, \
    # Z -- list of observations from one path, U -- list of controls from one path, \
    # A -- state transition matrix, B -- control coefficient matrix, C -- observation matrix, \
    # G -- control gain, K -- Kalman gain, N -- number of total time steps, \
    # d_X -- dimension of state, r -- scaling factor
    # computes dF/dK, where F = 1/2N *\sum_{n=0}^N (X_n^TX_n + rU_n^TU_n), with U_N=0
    # returns dF/dK (2-by-1)

    dXn = np.zeros((2, 2))
    dXHatn = np.zeros((2, 2))
    dUn = G @ dXHatn
    dF = np.zeros((1, 2))
    dF += X[0, :, :].T @ dXn + r * U[0, :, :].T @ dUn
    coef1 = A + B @ G
    coef2 = K @ C @ A
    for i in range(1, N+1):

        dXn_ = dXn
        dXn = A @ dXn_ + B @ dUn
        diag = Z[i, :, :] - C @ (A @ X_hat[i-1, :, :] + B @ U[i-1, :, :])
        diag = np.array([1] * d_X) * diag[0][0]
        dXHatn = coef1 @ dXHatn + np.diag(diag) + coef2 @ (dXn_ - dXHatn)
        dUn = G @ dXHatn
        dF += X[i, :, :].T @ dXn + r * U[i, :, :].T @ dUn

    return dF.T/N


def forward_G(X, X_hat, U, A, B, C, G, K, N, d_X, r):
    # X -- list of states from one path, X_hat -- list of state estimations from one path, \
    # U -- list of controls from one path, \
    # A -- state transition matrix, B -- control coefficient matrix, C -- observation matrix, \
    # G -- control gain, K -- Kalman gain, N -- number of total time steps, \
    # d_X -- dimension of state, r -- scaling factor
    # computes dF/dG, where F = 1/2N *\sum_{n=0}^N (X_n^TX_n + rU_n^TU_n), with U_N=0
    # returns dF/dG (1-by-2)

    dXn = np.zeros((2, 2))
    dXHatn = np.zeros((2, 2))
    dUn = X_hat[0, :, :].T + G @ dXHatn
    dF = np.zeros((1, 2))
    dF += X[0, :, :].T @ dXn + r * U[0, :, :].T @ dUn
    coef1 = np.identity(d_X) - K @ C
    coef2 = A + B @ G
    for i in range(1, N+1):

        dXn = A @ dXn + B @ dUn
        dXHatn = coef1 @ coef2 @ dXHatn + coef1 @ B @ X_hat[i-1, :, :].T + K @ C @ dXn
        dUn = X_hat[i, :, :].T + G @ dXHatn
        dF += X[i, :, :].T @ dXn + r * U[i, :, :].T @ dUn

    return dF/N
