# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program implements a simplified wrapper function for the optimization algorithms \
# which just returns the final K, G, and loss

import ultimate_optim
import stability_check
import numpy as np
import matplotlib.pyplot as plt


def wrapper(X0, A, C, B, G0, K0, N, S, R, d_X, d_Z, d_U, r, n, act, L, g, s_l, alpha, momentum, M, which):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G0 -- initial control gain, K0 -- initial Kalman gain, \
    # N -- number of total time steps, S -- observation noise covariance matrix, R -- system noise covariance matrix, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control, \
    # r -- scaling factor, n -- number of total gradient steps, \
    # act -- set to True to activate learning rate scheduler, \
    # L -- list of milestones, g -- multiplicative factor of learning rate decay,  s_l -- a list of random seeds, \
    # alpha -- learning rate, momentum -- momentum factor (can set to 0), M -- minibatch size, \
    # which -- name of the optimization algorithm to use (SGD, Adam, or RMSprop)
    # a wrapper function that calls the optimize in ultimate_optim.py for s in s_l

    is_stable = stability_check.check_stability(A, B, C, K0, G0, d_X)
    if is_stable is False:
        print("Dynamics is unstable. Choose another K0 and G0")
        return
    # print("Dynamics is stable")

    K_avg = np.zeros((2, 1))
    G_avg = np.zeros((1, 2))
    F_avg = np.zeros(n)

    for s in s_l:
        try:
            K, G, F_l, grad_K_l, grad_G_l = ultimate_optim.optimize(X0, A, C, B, G0, K0, N, S, R, d_X, d_Z, d_U, r, n, act, L,
                                                                    g, alpha, momentum, M, s, comp=None, which=which)
        except TypeError:
            return

        K_avg += K
        G_avg += G
        F_avg += F_l

    K_avg = K_avg/len(s_l)
    G_avg = G_avg/len(s_l)
    F_avg = F_avg/len(s_l)

    return K_avg, G_avg, F_avg[-1]

