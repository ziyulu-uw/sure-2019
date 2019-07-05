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
    # K0 -- initial Kalman gain, n -- number of total gradient steps, \
    # alpha -- learning rate, s_l -- a list of random seeds
    # a wrapper function that calls stochastic_gradient_descent(X0, A, C, N, R, S, K, n, alpha, s) for s in s_l \
    # and plots F vs n

    print("Initialization: K11 is {}, K12 is {}".format(K0[0][0], K0[1][0]))
    K_avg = np.array([[0.0], [0.0]])
    F_avg = np.zeros(n)

    # plots F vs n
    print("After {} iterations of stochastic gradient descent:".format(n))
    print("seed   K1      K2   InitialLoss FinalLoss   Gradient for each iteration")
    for s in s_l:
        K, F_l, grad_l = SGD.stochastic_gradient_descent(X0, A, C, N, R, S, K0, n, alpha, s)
        print("{}    {:.3f}    {:.3f}    {:.3f}    {:.3f}     ".format(s, K[0][0], K[1][0], F_l[0], F_l[-1]), grad_l)
        K_avg += K
        F_avg += F_l

    K_avg = K_avg/len(s_l)
    F_avg = F_avg/len(s_l)

    print("Averaging over {} random seeds, K11 is {:.3f}, K12 is {:.3f}. The final loss is {:.3f}".\
          format(len(s_l), K_avg[0][0], K_avg[1][0], F_avg[-1]))

    x = [i for i in range(n)]
    plt.plot(x, F_avg)
    plt.rcParams["axes.titlesize"] = 8
    plt.title("Stochastic gradient descent with {} steps and step size {}, averaged over {} random seeds".\
              format(str(n),str(alpha),str(len(s_l))))
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("mean squared error of one simulation")
    plt.show()

    return K_avg, F_avg
