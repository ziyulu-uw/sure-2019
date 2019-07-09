# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program does a convergence study of the finite difference approximation

import numpy as np
import matplotlib.pyplot as plt
import finite_diff_approx
import path_generation
import loss_gradient_computation


def conv_study(X0, A, C, N, dt, W, V, K, delta_K, m):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # N -- number of total time steps, dt -- step size, \
    # W -- sequence of system noise, V -- sequence of observation noise, K -- sequence of Kalman gains, \
    # delta_K -- difference in finite difference approximation, m -- index of Kalman gain to take derivative with
    # checks if the first order approximation works properly \
    # by plotting the norm of approximation error against the difference with log scaling on both axes
    # the plot should be linear

    X, Z = path_generation.path_generator(X0, A, C, N, W, V)
    X_hat = path_generation.filtered_path_generator(X0, A, C, K, Z, N)
    all_grad = loss_gradient_computation.compute_gradient(A, C, N, K, X, Z, X_hat)
    grad = all_grad[:, m]
    n = 10  # number of finite difference approximations to compute
    # print(np.linalg.norm(grad))
    print("direct gradient computation", grad)  # This is the gradient w.r.t. Km computed by formulas

    step_sizes = []  # delta K values
    errors = []  # norm of finite difference error

    print("----- finite difference approximation -----")
    print("  delta K       dF/dK_1         dF/dK_2      K_1 error  K_2 error error_norm  conv_factor")
    delta_K_n = delta_K
    for i in range(n):
        grad_approx = finite_diff_approx.finite_diff_approx(X0, A, C, N, W, V, K, m, delta_K_n)
        error_norm = np.linalg.norm(grad_approx - grad)
        if i == 0:
            print("{0:10.2e}  {1:14.6e}  {2:14.6e} {3:10.2e} {4:10.2e} {5:10.2e}".
                  format(delta_K_n, grad_approx[0],
                         grad_approx[1],
                         grad_approx[0] - grad[0],
                         grad_approx[1] - grad[1], error_norm))
        else:
            conv_factor = error_norm / errors[i - 1]
            print("{0:10.2e}  {1:14.6e}  {2:14.6e} {3:10.2e} {4:10.2e} {5:10.2e} {6:12.4e}".
                  format(delta_K_n, grad_approx[0],
                         grad_approx[1],
                         grad_approx[0] - grad[0],
                         grad_approx[1] - grad[1], error_norm, conv_factor))
        step_sizes.append(delta_K_n)
        errors.append(error_norm)
        delta_K_n = 2 * delta_K_n

    plt.loglog(step_sizes, errors, "o")
    plt.grid()
    plt.rcParams["axes.titlesize"] = 9
    plt.title("First order finite difference approximation of dF/dK_{} with K_{} = {}^T, N = {}, dt = {}".\
              format(m, m, K[:, m], N, dt))
    plt.xlabel("delta_K")
    plt.ylabel("Frobenius norm of (grad_approx - grad)")
    plt.show()
