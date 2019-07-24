# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program verifies the backward gradient computation using finite difference approximation

import Initialization as init
import path_generation as pgen
import gradient_error_computation as comp
import numpy as np
import matplotlib.pyplot as plt


def finite_diff_approx(K, delta_K, z):
    # K -- Kalman gain, delta_K -- difference in finite difference approximation, z -- random seed
    # performs first order finite difference approximation of dF/dK: dF/dK = (F(K+delta_K) - F(K))/delta_K
    # returns the finite difference approximation of dF/dK, and F

    grad_approx = np.array([[0.0, 0.0]])

    # compute F(K)
    np.random.seed(z)  # set random seed
    X = init.X0  # initial state
    X_hat = init.X0  # initial state estimate
    sum_errors = 0

    for n in range(init.N):
        Z_hat = init.C @ init.A @ X_hat  # predicted observation
        W = np.random.multivariate_normal([0, 0], init.R)  # gaussian system noise with mean 0 covariance R
        W = np.array(W, ndmin=2)
        W = np.transpose(W)
        X = init.A @ X + W  # state update
        V = np.random.normal(0, init.Q)  # gaussian observation noise with mean 0 variance Q
        Z = init.C @ X + V  # observation
        X_hat = init.A @ X_hat + K * (Z - Z_hat)  # state estimate
        error = X_hat - X
        sum_errors += error[0][0] ** 2 + error[1][0] ** 2

    F = sum_errors/(2*init.N)

    # compute F(K+delta_K)
    for i in range(len(K)):

        np.random.seed(z)  # set random seed
        X = init.X0  # initial state
        X_hat = init.X0  # initial state estimate
        K[i][0] += delta_K
        sum_errors = 0

        for n in range(init.N):
            Z_hat = init.C @ init.A @ X_hat  # predicted observation
            W = np.random.multivariate_normal([0, 0], init.R)  # gaussian system noise with mean 0 covariance R
            W = np.array(W, ndmin=2)
            W = np.transpose(W)
            X = init.A @ X + W  # state update
            V = np.random.normal(0, init.Q)  # gaussian observation noise with mean 0 variance Q
            Z = init.C @ X + V  # observation
            X_hat = init.A @ X_hat + K * (Z - Z_hat)  # state estimate
            error = X_hat - X
            sum_errors += error[0][0] ** 2 + error[1][0] ** 2

        F_ = sum_errors/(2*init.N)
        grad_approx[0][i] = (F_ - F)/delta_K
        K[i][0] -= delta_K

    return grad_approx, F


def check_order(K, delta_K, z, n):
    # K -- Kalman gain, delta_K -- difference in finite difference approximation, \
    # z -- random seed, n -- number of multiples of delta_K
    # checks if the above first order approximation works properly \
    # by plotting the norm of approximation error against the difference
    # the plot should be linear

    x_L = [0]
    y_L = [0]
    L_state, L_obs, L_est = pgen.generate_path(K, 1)
    grad = comp.compute_gradient(K, L_state, L_obs, L_est)
    # print(np.linalg.norm(grad))
    print("direct gradient computation", grad)  # This is the gradient computed by formulas
    for i in range(n):
        delta_K_n = delta_K * (i+1)
        grad_approx, F = finite_diff_approx(K, delta_K_n, z)
        print("gradient approximation with delta_K = {}".format(delta_K_n),grad_approx)  # This is the gradient given by finite difference approximation
        x_L.append(delta_K_n)
        y_L.append(np.linalg.norm(grad_approx - grad))

    norm1 = np.linalg.norm(grad)
    norm2 = y_L[1]
    print("norm of smallest difference/norm of gradient", norm2/norm1)
    plt.plot(x_L, y_L)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.rcParams["axes.titlesize"] = 10
    plt.title("First order finite difference approximation of dF/dK with K = {}, N = {}, dt = {}".format(K, init.N, init.dt))
    plt.xlabel("delta_K")
    plt.ylabel("Frobenius norm of (grad_approx - grad)")
    plt.show()
