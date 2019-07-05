# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program does a convergence study of the finite difference approximation

import numpy as np
import matplotlib.pyplot as plt
import finite_diff_approx
import path_generation
import loss_gradient_computation


def conv_study(X0, A, C, N, dt, W_l, V_l, K, delta_K, n):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # N -- number of total time steps, dt -- step size, \
    # W_l -- sequence of system noise, V_l -- sequence of observation noise, \
    # K -- Kalman gain, delta_K -- difference in finite difference approximation, n -- number of multiples of delta_K
    # checks if the first order approximation works properly \
    # by plotting the norm of approximation error against the difference
    # the plot should be linear

    X_l, Z_l = path_generation.path_generator(X0, A, C, N, W_l, V_l)
    X_hat_l = path_generation.filtered_path_generator(X0, A, C, K, Z_l, N)
    grad = loss_gradient_computation.compute_gradient(A, C, N, K, X_l, Z_l, X_hat_l)
    # print(np.linalg.norm(grad))
    print("direct gradient computation", grad)  # This is the gradient computed by formulas

    x_L = [0]
    y_L = [0]
    print("----- finite difference approximation -----")
    print("  delta K       dF/dK_1         dF/dK_2      K_1 error  K_2 error")
    for i in range(n):
        delta_K_n = delta_K * (i+1)
        grad_approx = finite_diff_approx.finite_diff_approx(X0, A, C, N, W_l, V_l, K, delta_K_n)
        print("{0:10.2e}  {1:14.6e}  {2:14.6e} {3:10.2e} {4:10.2e}".
              format(delta_K_n, grad_approx[0][0], grad_approx[0][1], grad_approx[0][0] - grad[0][0],
                     grad_approx[0][1] - grad[0][1]))
        x_L.append(delta_K_n)
        y_L.append(np.linalg.norm(grad_approx - grad))

    norm1 = np.linalg.norm(grad)
    norm2 = y_L[1]
    print("norm of smallest difference/norm of gradient", norm2/norm1)
    plt.plot(x_L, y_L)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.rcParams["axes.titlesize"] = 10
    plt.title("First order finite difference approximation of dF/dK with K = {}, N = {}, dt = {}".format(K, N, dt))
    plt.xlabel("delta_K")
    plt.ylabel("Frobenius norm of (grad_approx - grad)")
    plt.show()
