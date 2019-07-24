# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program computes the gradient of F w.r.t. K and G using finite difference approximation

import numpy as np
import path_generation
import loss_computation


def finite_diff_approx_K(X0, A, C, B, G, K, N, r, W, V, d_X, d_Z, d_U, delta_K):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G -- control gain, K -- Kalman gain, \
    # N -- number of total time steps, r -- scaling factor
    # W -- sequence of system noise, V -- sequence of observation noise, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control
    # delta_K -- difference in finite difference approximation
    # performs first order finite difference approximation of dF/dK: dF/dK = (F(K+delta_K) - F(K))/delta_K
    # returns the finite difference approximation of dF/dK

    grad_approx_K = np.zeros((1, 2))

    # compute F(K)
    X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
    F = loss_computation.compute_loss(X, U, N, r)

    # compute F(K+delta_K)
    for i in range(len(K)):

        K[i, 0] += delta_K
        X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
        F_ = loss_computation.compute_loss(X, U, N, r)
        grad_approx_K[0, i] = (F_ - F)/delta_K
        K[i, 0] -= delta_K

    return grad_approx_K


def finite_diff_approx_G(X0, A, C, B, G, K, N, r, W, V, d_X, d_Z, d_U, delta_G):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G -- control gain, K -- Kalman gain, \
    # N -- number of total time steps, r -- scaling factor
    # W -- sequence of system noise, V -- sequence of observation noise, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control
    # delta_G -- difference in finite difference approximation
    # performs first order finite difference approximation of dF/dG: dF/dG = (F(G+delta_G) - F(G))/delta_G
    # returns the finite difference approximation of dF/dG

    grad_approx_G = np.zeros((1, 2))

    # compute F(G)
    X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
    F = loss_computation.compute_loss(X, U, N, r)

    # compute G(G+delta_G)
    for i in range(len(G[0])):
        G[0, i] += delta_G
        X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
        F_ = loss_computation.compute_loss(X, U, N, r)
        grad_approx_G[0, i] = (F_ - F) / delta_G
        G[0, i] -= delta_G

    return grad_approx_G
