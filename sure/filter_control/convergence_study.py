# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program does a convergence study of the finite difference approximation

import numpy as np
import matplotlib.pyplot as plt
import finite_diff_approx
import path_generation
import backward_grad
import forward_grad


def conv_study_K(X0, A, C, B, G, K, N, r, W, V, d_X, d_Z, d_U, delta_K, which):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G -- control gain, K -- Kalman gain, \
    # N -- number of total time steps, r -- scaling factor
    # W -- sequence of system noise, V -- sequence of observation noise, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control
    # delta_K -- difference in finite difference approximation
    # which -- which gradient computation method to use (F: forward; B: backward)
    # checks if the first order approximation works properly \
    # by plotting the norm of approximation error against the difference with log scaling on both axes
    # the plot should be linear

    X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
    if which == 'B':
        grad_K, grad_G = backward_grad.compute_gradient(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X)
    elif which == 'F':
        grad_K = forward_grad.forward_K(X, X_hat, Z, U, A, B, C, G, K, N, d_X, r)
    else:
        print("Invalid argument")

    grad_K = grad_K.transpose()
    n = 10  # number of finite difference approximations to compute
    print("direct gradient computation", grad_K)  # This is the gradient computed by formulas

    # convergence study in K derivative
    step_sizes = []  # delta K values
    errors = []  # norm of finite difference error

    print("----- finite difference approximation -----")
    print("  delta_K_n      dF/dK_1         dF/dK_2     K_1 error  K_2 error error_norm  conv_factor")
    delta_K_n = delta_K
    for i in range(n):
        grad_approx = finite_diff_approx.finite_diff_approx_K(X0, A, C, B, G, K, N, r, W, V, d_X, d_Z, d_U, delta_K_n)
        error_norm = np.linalg.norm(grad_approx - grad_K)
        if i == 0:
            print("{0:10.2e}  {1:14.6e}  {2:14.6e} {3:10.2e} {4:10.2e} {5:10.2e}".
                  format(delta_K_n, grad_approx[0][0],
                         grad_approx[0][1],
                         grad_approx[0][0] - grad_K[0][0],
                         grad_approx[0][1] - grad_K[0][1], error_norm))
        else:
            conv_factor = error_norm / errors[i - 1]
            print("{0:10.2e}  {1:14.6e}  {2:14.6e} {3:10.2e} {4:10.2e} {5:10.2e} {6:12.4e}".
                  format(delta_K_n, grad_approx[0][0],
                         grad_approx[0][1],
                         grad_approx[0][0] - grad_K[0][0],
                         grad_approx[0][1] - grad_K[0][1], error_norm, conv_factor))
        step_sizes.append(delta_K_n)
        errors.append(error_norm)
        delta_K_n = 2 * delta_K_n

    plt.loglog(step_sizes, errors, "o")
    plt.grid()
    plt.rcParams["axes.titlesize"] = 10
    plt.title("First order finite difference approximation of dF/dK with K = {}".format(K))
    plt.xlabel("delta_K")
    plt.ylabel("Frobenius norm of (grad_approx - grad)")
    plt.show()


def conv_study_G(X0, A, C, B, G, K, N, r, W, V, d_X, d_Z, d_U, delta_G, which):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G -- control gain, K -- Kalman gain, \
    # N -- number of total time steps, r -- scaling factor
    # W -- sequence of system noise, V -- sequence of observation noise, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control
    # delta_G -- difference in finite difference approximation
    # which -- which gradient computation method to use (F: forward; B: backward)
    # checks if the first order approximation works properly \
    # by plotting the norm of approximation error against the difference with log scaling on both axes
    # the plot should be linear

    X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
    if which == 'B':
        grad_K, grad_G = backward_grad.compute_gradient(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X)
    elif which == 'F':
        grad_G = forward_grad.forward_G(X, X_hat, U, A, B, C, G, K, N, d_X, r)
    else:
        print("Invalid argument")

    n = 10  # number of finite difference approximations to compute
    print("direct gradient computation", grad_G)  # This is the gradient computed by formulas

    # convergence study in K derivative
    step_sizes = []  # delta K values
    errors = []  # norm of finite difference error

    print("----- finite difference approximation -----")
    print("  delta_G_n      dF/dG_1         dF/dG_2     G_1 error  G_2 error error_norm  conv_factor")
    delta_G_n = delta_G
    for i in range(n):
        grad_approx = finite_diff_approx.finite_diff_approx_G(X0, A, C, B, G, K, N, r, W, V, d_X, d_Z, d_U, delta_G_n)
        error_norm = np.linalg.norm(grad_approx - grad_G)
        if i == 0:
            print("{0:10.2e}  {1:14.6e}  {2:14.6e} {3:10.2e} {4:10.2e} {5:10.2e}".
                  format(delta_G_n, grad_approx[0][0],
                         grad_approx[0][1],
                         grad_approx[0][0] - grad_G[0][0],
                         grad_approx[0][1] - grad_G[0][1], error_norm))
        else:
            conv_factor = error_norm / errors[i - 1]
            print("{0:10.2e}  {1:14.6e}  {2:14.6e} {3:10.2e} {4:10.2e} {5:10.2e} {6:12.4e}".
                  format(delta_G_n, grad_approx[0][0],
                         grad_approx[0][1],
                         grad_approx[0][0] - grad_G[0][0],
                         grad_approx[0][1] - grad_G[0][1], error_norm, conv_factor))
        step_sizes.append(delta_G_n)
        errors.append(error_norm)
        delta_G_n = 2 * delta_G_n

    plt.loglog(step_sizes, errors, "o")
    plt.grid()
    plt.rcParams["axes.titlesize"] = 10
    plt.title("First order finite difference approximation of dF/dG with G = {}".format(G))
    plt.xlabel("delta_G")
    plt.ylabel("Frobenius norm of (grad_approx - grad)")
    plt.show()
