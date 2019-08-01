# Author: Ziyu Lu, Xinyu Li
# Email: zl1546@nyu.edu, xl1796@nyu.edu
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


def transpose(x):
    # takes transpose of each matrix inside a tensor x (n*d*c --> n*c*d)
    d = len(x[0, :, 0])  # num of rows of each matrix
    c = len(x[0, 0, :])  # num of columns of each matrix
    n = len(x[:, 0, 0])  # num of matrices in the tensor x
    x_T = np.zeros([n, c, d])  # allocate space
    for i in range(d):
        x_T[:, :, i] = x[:, i, :]
    return x_T


def compute_multi_gradients(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X):
    # compute gradients for multiple paths

    m = len(X)
    P = np.zeros((m, 1, 2))
    T = 2*transpose(X[:, N, :, :])
    Q = np.zeros((m, 1, 2))
    H = np.zeros((m, 1, 2))

    for i in range(N - 1, -1, -1):  # move backward

        x_hat = X_hat[:, i, :, :]  # some frequently used variables
        x_hat_transpose = transpose(x_hat)
        u = U[:, i, :, :]
        diag = Z[:, i + 1, :, :] - C @ (A @ x_hat + B @ u)
        diag_matrices = np.zeros((m, 2, 2))
        for j in range(m):
            diag_array = np.array([1] * d_X) * diag[j][0][0]
            diag_matrix = np.diag(diag_array)
            diag_matrices[j] = diag_matrix

        Q = Q + P @ diag_matrices
        H = 2 * r * (transpose(u) @ x_hat_transpose) + H \
            + P @ (np.identity(d_X) - K @ C) @ B @ x_hat_transpose + T @ B @ x_hat_transpose
        P_ = P
        T_ = T
        P = 2 * r * (transpose(u) @ G) + P_ @ ((np.identity(d_X) - K @ C) @ A + B @ G) + T_ @ B @ G
        T = 2 * transpose(X[:, i, :, :]) + P_ @ K @ C @ A + T_ @ A

    return transpose(Q) / (2 * N), H / (2 * N)

# from initialization import *
# import noise_generation
# import path_generation
# m = 3
# N = 4
# K = np.array([[0.5], [0.5]])
# G = np.array([[-1.0, -0.1]])
# W = noise_generation.vectorized_system_noise_generator(m, d_X, N, R)
# V = noise_generation.vectorized_observation_noise_generator(m, d_Z, N, S)
# X1, Z1, U1, X_hat1 = path_generation.path_generator(X0, A, C, B, G, K, N, W[1], V[1], d_X, d_Z, d_U)
# Q1, H1 = compute_gradient(A, C, B, G, K, N, X1, Z1, U1, X_hat1, r, d_X)
# print(Q1, H1)
# X, Z, U, X_hat = path_generation.multi_paths_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
# Q, H = compute_multi_gradients(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X)
# print(Q[1], H[1])

