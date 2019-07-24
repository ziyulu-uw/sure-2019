# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program computes the gradients of F w.r.t. K and G for a given path using backward propagation

import numpy as np


def compute_gradient(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X):
    # A -- state transition matrix, C -- observation matrix, B -- control coefficient matrix, \
    # G -- control gain, K -- Kalman gain, N -- number of total time steps, \
    # X -- list of states from one path, Z -- list of observations from one path, \
    # U -- list of controls from one path, X_hat -- list of state estimations from one path, \
    # r -- scaling factor, d_X -- dimension of state
    # computes dF/dK, dF/dG, where F = 1/2N *\sum_{n=0}^N (X_n^TX_n + rU_n^TU_n), with U_N=0
    # returns dF/dK, dF/dG

    P = np.zeros((1, 2))
    T = 2 * X[N, :, :].transpose()
    Q = np.zeros((1, 2))
    H = np.zeros((1, 2))

    for i in range(N - 1, -1, -1):  # move backward

        x_hat = X_hat[i, :, :]  # some frequently used variables
        x_hat_transpose = x_hat.transpose()
        u = U[i, :, :]

        diag = Z[i + 1, :, :] - C @ (A @ x_hat + B @ u)
        diag = np.array([1] * d_X) * diag[0][0]
        Q = Q + P @ np.diag(diag)
        H = 2 * r * (u.transpose() @ x_hat_transpose) + H \
            + P @ (np.identity(d_X) - K @ C) @ B @ x_hat_transpose + T @ B @ x_hat_transpose
        P_ = P
        T_ = T
        P = 2 * r * (u.transpose() @ G) + P_ @ ((np.identity(d_X) - K @ C) @ A + B @ G) + T_ @ B @ G
        T = 2 * X[i, :, :].transpose() + P_ @ K @ C @ A + T_ @ A

    return Q.transpose() / (2 * N), H / (2 * N)
