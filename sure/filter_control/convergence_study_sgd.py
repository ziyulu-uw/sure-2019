# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program compares the numerical result with the theoretical result

import numpy as np
import LQG  # simulation with LQG is done here
from init_func import init  # initial parameter are set using the func in this module
from LQG_tool import generate_noise, Plot_K, Plot_G, Plot_X  # noise generation
from LQG_loss_computation import compute_cost, after_train_cost
import ultimate_wrappers
import matplotlib.pyplot as plt
import performance_test


def LQG_sol():
    k = 1  # spring constant
    gamma = 0.1  # friction coefficient
    sigma = 0.1  # noise coefficient in SDE
    x0 = 1  # initial condition
    v0 = 0
    r = 1  # scaling factor in the cost
    obv_noise = 0.3  # covariance of observation noise
    t1 = 60
    dt = 1.5  # Please choose what can divide t1
    n = 1000  # number of paths
    x0, v0, t, X0, A, B, C, S, R, d_X, d_U, d_Z, r, N = init(k, gamma, dt, sigma, x0, v0, r, obv_noise, t1)

    # -------------------------------- Find Filter and Control using LQG-------------------------------------------
    # Find out Sn (a series of matrices that will be used in Gn calculation)
    Sn_list = LQG.Sn_backward(A, B, r, n, N)

    # Generate process and observation noise
    W = generate_noise(R, n, N + 1)  # process noise: R-cov  [Wn is not used in the simulation]
    V = generate_noise(S, n, N + 1)  # observation noise: S-cov

    # Simulate the process with LQG control
    K_val, G_val, X_val, TrueX_val = LQG.LQG_simulation(x0, v0, A, B, C, R, S, r, Sn_list, W, V, n, N, DoLQG=True)
    cost = compute_cost(X_val, TrueX_val, G_val, N, r)
    K = np.reshape(np.average(K_val, axis=0)[-1, :], [2, 1])
    G = np.reshape(G_val[-1, :], [1, 2])

    # Calculate the after train cost
    G_val_ = np.zeros([N + 1, 2])
    G_val_[:, 0] = G[0, 0]  # in this case, G_val_ from t=0 to t=N should have same G1 and G2 values
    G_val_[:, 1] = G[0, 1]
    X_val_, TrueX_val_ = after_train_cost(K, G, W, V, A, B, C, x0, v0, n, N)
    cost2 = compute_cost(X_val_, TrueX_val_, G_val_, N, r)

    # Plot_X(X_val, t)
    # Plot_K(K_val, t)
    # Plot_G(G_val, t)

    print("---- Theoretical results ----")
    print("Cost:{:10.2e}".format(cost))
    print("Test cost:{:10.2e}".format(cost2))
    print("Steady state K:[{:10.2e},{:10.2e}]".format(K[0][0], K[1][0]))
    print("Steady state G:[{:10.2e},{:10.2e}]".format(G[0][0], G[0][1]))
    print('\n')

    return cost, cost2, K, G


def SGD_sol(cost2, bestK, bestG):
    # To compare the theoretical and numerical results, their setup should be the same
    k = 1  # spring constant
    gamma = 0.1  # friction coefficient
    sigma = 0.1  # noise coefficient in SDE
    x0 = 1  # initial condition
    v0 = 0
    r = 1  # scaling factor in the cost
    obv_noise = 0.3  # covariance of observation noise
    t1 = 30
    dt = 1.5  # Please choose what can divide t1
    x0, v0, t, X0, A, B, C, S, R, d_X, d_U, d_Z, r, N = init(k, gamma, dt, sigma, x0, v0, r, obv_noise, t1)

    K = np.array([[0.5], [0.5]])  # initial Kalman gain
    G = np.array([[-1.0, -0.1]])  # initial control gain
    comp = [bestK, bestG, cost2]
    print("---- Numerical results ----")
    K_avg, G_avg, F_avg, diff_K_avg, diff_K_avg = ultimate_wrappers.wrapper(X0, A, C, B, G, K, N, S, R, d_X, d_Z, d_U, r, n=200, act=False, L=[2000, 3000], g=0.1,
                                                                            s_l=[1], alpha=0.1, betas=0.25, momentum=0, M=8, avg=None, comp=comp, which='RMSprop', zoom=None)
    test_loss = performance_test.test(X0, A, C, B, G_avg, K_avg, N, R, S, r, d_X, d_Z, d_U, n=1000, disp=True)

    return K_avg, G_avg, test_loss

cost, cost2, K, G = LQG_sol()
SGD_sol(cost2, K, G)