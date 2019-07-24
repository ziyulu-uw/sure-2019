# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program implements a wrapper function for stochastic gradient descent \
# which averages the results over a list of random seeds and plots the results

import SGD
import numpy as np
import matplotlib.pyplot as plt


def Stochastic_gradient_descent(X0, A, C, N, R, S, K0, n, alpha, s_l):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # N -- number of total time steps, R -- system noise covariance matrix, S -- observation noise covariance matrix, \
    # K0 -- sequence of initial Kalman gains, n -- number of total gradient steps, \
    # alpha -- learning rate, s_l -- a list of random seeds
    # a wrapper function that calls stochastic_gradient_descent(X0, A, C, N, R, S, K, n, alpha, s) for s in s_l \
    # and plots F vs n

    d_X = len(X0)  # dimension of state
    np.set_printoptions(precision=2, linewidth=2000)
    print("------ initialization ------")
    print(K0)
    print('\n')

    K_avg = np.zeros((d_X, N+1))
    F_avg = np.zeros(n)

    # plots F vs n
    print("------ after {} iterations of stochastic gradient descent ------".format(n))
    for s in s_l:
        K, F_l, grad_l = SGD.stochastic_gradient_descent(X0, A, C, N, R, S, K0, n, alpha, s)
        print("seed {:2d}    initial loss ={:10.2e}   final loss ={:10.2e}    largest gradient ={:10.2e}".\
              format(s, F_l[0], F_l[-1], np.amax(grad_l)))
        print(K)
        print('\n')

        K_avg += K
        F_avg += F_l

    K_avg = K_avg/len(s_l)
    F_avg = F_avg/len(s_l)

    print("------ averaging over {} random seeds ------".format(len(s_l)))
    print("final loss ={:10.2e}".format(F_avg[-1]))
    print(K_avg)

    x = [i for i in range(n)]
    plt.plot(x, F_avg)
    plt.rcParams["axes.titlesize"] = 8
    plt.title("Stochastic gradient descent with {} steps and step size {}, averaged over {} random seeds".\
              format(str(n),str(alpha),str(len(s_l))))
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("mean squared error of one simulation")
    plt.show()

    return K_avg, F_avg
