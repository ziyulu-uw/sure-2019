# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program implements a wrapper function for the optimization algorithms \
# which averages the results over a list of random seeds and plots the results

import optimization
import stability_check
import numpy as np
import matplotlib.pyplot as plt


def wrapper(X0, A, C, B, G0, K0, N, S, R, d_X, d_Z, d_U, r, n, L, g, s_l, which, alpha, momentum=0):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G0 -- initial control gain, K0 -- initial Kalman gain, \
    # N -- number of total time steps, S -- observation noise covariance matrix, R -- system noise covariance matrix, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control, \
    # r -- scaling factor, n -- number of total gradient steps, \
    # # L -- list of milestones, g -- multiplicative factor of learning rate decay,  s_l -- a list of random seeds, \
    # which -- name of the optimization algorithm to use (SGD, Adam, or RMSprop), \
    # alpha -- learning rate, momentum -- optional momentum factor for SGD
    # a wrapper function that calls one of the optimization methods in optimization.py for s in s_l \
    # and plots F vs n

    is_stable = stability_check.check_stability(A, B, C, K0, G0, d_X)
    if is_stable is False:
        print("Dynamics is unstable. Choose another K0 and G0")
        return
    print("Dynamics is stable")

    print("Optimizing using {} algorithm".format(which))
    print("---- Initialization ----")
    print("K11 is {}, K12 is {}".format(K0[0][0], K0[1][0]))
    print("G1 is {}, G2 is {}".format(G0[0][0], G0[0][1]))
    K_avg = np.zeros((2, 1))
    G_avg = np.zeros((1, 2))
    F_avg = np.zeros(n)

    # plots F vs n
    print("---- After {} iterations ----".format(n))
    print("seed      K1            K2         G1         G2       InitialLoss   FinalLoss   Max_K_Gradient  Max_G_Gradient")
    for s in s_l:
        if which == 'SGD':
            K, G, F_l, grad_K_l, grad_G_l = optimization.SGD(X0, A, C, B, G0, K0, N, S, R, d_X, d_Z, d_U, r, n, L, g, momentum, alpha, s)
        elif which == 'Adam':
            K, G, F_l, grad_K_l, grad_G_l = optimization.Adam(X0, A, C, B, G0, K0, N, S, R, d_X, d_Z, d_U, r, n, L, g, alpha, s)
        elif which == 'RMSprop':
            K, G, F_l, grad_K_l, grad_G_l = optimization.RMSprop(X0, A, C, B, G0, K0, N, S, R, d_X, d_Z, d_U, r, n, L, g, alpha, s)
        else:
            print('Invalid algorithm')
            break

        print("{:2d}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}   {:10.2e}     {:10.2e}     {:10.2e}"\
              .format(s, K[0][0], K[1][0], G[0][0], G[0][1], F_l[0], F_l[-1], np.amax(grad_K_l), np.amax(grad_G_l)))
        K_avg += K
        G_avg += G
        F_avg += F_l

    K_avg = K_avg/len(s_l)
    G_avg = G_avg/len(s_l)
    F_avg = F_avg/len(s_l)

    print("---- Averaging over {} random seeds ----".format(len(s_l)))
    print("K11 is{:10.2e}, K12 is{:10.2e}".format(K_avg[0][0], K_avg[1][0]))
    print("G1 is{:10.2e}, G2 is{:10.2e}".format(G_avg[0][0], G_avg[0][1]))
    print("The final loss is{:10.2e}".format(F_avg[-1]))

    x = [i for i in range(n)]
    plt.plot(x, F_avg)
    plt.yscale("log")
    plt.rcParams["axes.titlesize"] = 8
    plt.title("{} algorithm with {} steps and step size {}, averaged over {} random seeds".\
              format(which, str(n),str(alpha),str(len(s_l))))
    plt.xlabel("number of optimization steps")
    plt.ylabel("mean squared error of one simulation")
    plt.show()

    return K_avg, G_avg, F_avg
