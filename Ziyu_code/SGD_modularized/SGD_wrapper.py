# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program implements a wrapper function for stochastic gradient descent \
# which averages the results over a list of random seeds and plots the results

import SGD
import numpy as np
import matplotlib.pyplot as plt


def Stochastic_gradient_descent(K0, n,alpha, z_l):
    # z_l -- a list of random seeds
    # a wrapper function that calls stochastic_gradient_descent(K, n, alpha, z) for z in z_l \
    # and plots F vs n

    print("Initialization: K11 is {}, K12 is {}".format(K0[0][0], K0[1][0]))
    K_avg = np.array([[0.0], [0.0]])
    err_avg = np.zeros(n)

    # plots F vs n
    for z in z_l:
        K, err_L, grad_L = SGD.stochastic_gradient_descent(K0, n, alpha, z)
        print("Seed {}: After {:d} iterations, K11 is {:.3f}, K12 is {:.3f}. The final loss is {:.3f}".\
              format(z, n, K[0][0], K[1][0], err_L[-1]))
        print("Gradient for each iteration", grad_L)
        K_avg += K
        err_avg += err_L

    K_avg = K_avg/len(z_l)
    err_avg = err_avg/len(z_l)

    print("Averaging over {} random seeds, K11 is {:.3f}, K12 is {:.3f}. The final loss is {:.3f}".\
          format(len(z_l), K_avg[0][0], K_avg[1][0], err_avg[-1]))

    x = [i for i in range(n)]
    plt.plot(x, err_avg)
    plt.title("Stochastic gradient descent with {} steps and step size {}, averaged over {} random seeds".\
              format(str(n),str(alpha),str(len(z_l))))
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("mean squared error of one simulation")
    plt.show()

    return K_avg, err_avg

