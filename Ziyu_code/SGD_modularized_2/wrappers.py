# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program implements a wrapper function for the optimization algorithms \
# which averages the results over a list of random seeds and plots the results

import optimization
import numpy as np
import matplotlib.pyplot as plt


def wrapper(X0, A, C, N, R, S, K0, n, s_l, which, alpha, momentum=0):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # N -- number of total time steps, R -- system noise covariance matrix, S -- observation noise covariance matrix, \
    # K0 -- transpose of initial Kalman gain, n -- number of total gradient steps, s_l -- a list of random seeds, \
    # which -- name of the optimization algorithm to use (SGD, Adam, or RMSprop), \
    # alpha -- learning rate, momentum -- optional momentum factor for SGD
    # a wrapper function that calls one of the optimization methods in optimization.py for s in s_l \
    # and plots F vs n

    print("Optimizing using {} algorithm".format(which))
    print("Initialization: K11 is {}, K12 is {}".format(K0[0], K0[1]))
    K_avg = np.array([0.0, 0.0])
    F_avg = np.zeros(n)

    # plots F vs n
    print("After {} iterations:".format(n))
    print("seed      K1            K2         InitialLoss    FinalLoss  First_10_Gradients")
    for s in s_l:
        if which == 'SGD':
            K, F_l, grad_l = optimization.SGD(X0, A, C, N, R, S, K0, n, momentum, alpha, s)
        elif which == 'Adam':
            K, F_l, grad_l = optimization.Adam(X0, A, C, N, R, S, K0, n, alpha, s)
        elif which == 'RMSprop':
            K, F_l, grad_l = optimization.RMSprop(X0, A, C, N, R, S, K0, n, alpha, s)
        else:
            print('Invalid algorithm')
            break

        grad_to_print = ""
        for i in range(10):
            grad_to_print += "[{:10.2e} {:10.2e}]  ".format(grad_l[i][0], grad_l[i][1])
        print("{:2d}    {:10.2e}    {:10.2e}    {:10.2e}    {:10.2e}   ".format(s, K[0], K[1], F_l[0], F_l[-1])\
              + grad_to_print)
        K_avg += K
        F_avg += F_l

    K_avg = K_avg/len(s_l)
    F_avg = F_avg/len(s_l)

    print("Averaging over {} random seeds, K11 is{:10.2e}, K12 is{:10.2e}. The final loss is{:10.2e}".\
          format(len(s_l), K_avg[0], K_avg[1], F_avg[-1]))

    x = [i for i in range(n)]
    plt.plot(x, F_avg)
    plt.rcParams["axes.titlesize"] = 8
    plt.title("{} algorithm with {} steps and step size {}, averaged over {} random seeds".\
              format(which, str(n),str(alpha),str(len(s_l))))
    plt.xlabel("number of optimization steps")
    plt.ylabel("mean squared error of one simulation")
    plt.show()

    return K_avg, F_avg
