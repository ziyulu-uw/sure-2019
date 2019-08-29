# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: August 2019
# Description: This program generates plot with input data (cost at each gradient descent iteration, etc)

import numpy as np
import matplotlib.pyplot as plt


def multi_plot(data, p, which, nIter, alpha, M, betas, label=None):

    if p == 1:  # when plotting filter_l, control_l, gradF_l, or gradC_l
        n = len(data[0])
        x = [i for i in range(n)]
        plt.figure(1)
        for i in range(len(data)):
            plt.plot(x, data[i], label='{}{}'.format(label, i+1))

        # plt.yscale("log")
        plt.rcParams["axes.titlesize"] = 8
        plt.title("{} algorithm with {} steps, step size {}, minibatch size {}, smoothing constant {}".
                  format(which, nIter, alpha, M, betas))
        plt.xlabel("number of optimization steps")
        plt.ylabel("filter parameters")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif p == 2:  # when plotting the cost of a single trajectory
        n = len(data)
        x = [i for i in range(n)]
        plt.figure(1)
        plt.plot(x, data)
        plt.yscale("log")
        plt.rcParams["axes.titlesize"] = 8
        plt.title("{} algorithm with {} steps, step size {}, minibatch size {}, smoothing constant {}".
                  format(which, nIter, alpha, M, betas))
        plt.xlabel("number of optimization steps")
        plt.ylabel("cost")
        # plt.legend()
        plt.grid(True)
        plt.show()

    elif p == 3:  # when plotting costs of multiple trajectories
        n = len(data[0])
        x = [i for i in range(n)]
        plt.figure(1)
        for i in range(len(data)):
            plt.plot(x, data[i], label='Cost: M={}'.format(label[i]))

        plt.yscale("log")
        plt.rcParams["axes.titlesize"] = 8
        plt.title("{} algorithm with {} steps, step size {}, minibatch size {}, smoothing constant {}".
                  format(which, nIter, alpha, M, betas))
        plt.xlabel("number of optimization steps")
        plt.ylabel("cost")
        plt.legend()
        plt.grid(True)
        plt.show()


all_data = np.load('out.npz')
print(sorted(all_data.files))
# print(all_data['Filter'])
data = all_data['filter_l']

multi_plot(data, p=1, which='RMSprop', nIter=10, alpha=1e-3, M=1, betas=0.9, label='K')
