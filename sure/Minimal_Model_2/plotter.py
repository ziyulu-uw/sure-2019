# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: August 2019
# Description: This program generates plot with input data (cost at each gradient descent iteration, etc)

import numpy as np
import matplotlib.pyplot as plt


def multi_plot(data, p, which, nIter, alpha, M, betas, log, label=None):

    if p == 1:  # when plotting filter_l, control_l, gradF_l, or gradC_l
        n = len(data[0])
        x = [i for i in range(n)]
        plt.figure(1)
        for i in range(len(data)):
            plt.plot(x, data[i], label='{}{}'.format(label, i+1))
        if log is True:
            plt.yscale("log")
        plt.rcParams["axes.titlesize"] = 8
        plt.title("{} algorithm with {} steps, step size {}, minibatch size {}, smoothing constant {}".
                  format(which, nIter, alpha, M, betas))
        plt.xlabel("number of optimization steps")
        if label == 'K':
            plt.ylabel("filter parameters")
        elif label == 'H':
            plt.ylabel("control parameters")
        elif label == 'grad K':
            plt.ylabel("gradients of filter parameters")
        elif label == 'grad H':
            plt.ylabel("gradients of control parameters")

        plt.legend()
        plt.grid(True)
        plt.show()

    elif p == 2:  # when plotting the cost of a single trajectory
        n = len(data)
        x = [i for i in range(n)]
        plt.figure(1)
        plt.plot(x, data)
        if log is True:
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

        if log is True:
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
cost_l = all_data['cost_l']
print(cost_l[-1])
filter_l = all_data['filter_l']
control_l = all_data['control_l']
gradF_l = all_data['gradF_l']
gradC_l = all_data['gradC_l']


multi_plot(cost_l, p=2, which='RMSprop', nIter=1000, alpha=1e-5, M=1, betas=0.99, log=True, label=None)
multi_plot(filter_l, p=1, which='RMSprop', nIter=1000, alpha=1e-5, M=1, betas=0.99, log=False, label='K')
multi_plot(control_l, p=1, which='RMSprop', nIter=1000, alpha=1e-5, M=1, betas=0.99, log=False, label='H')
multi_plot(gradF_l, p=1, which='RMSprop', nIter=1000, alpha=1e-5, M=1, betas=0.99, log=False, label='grad K')
multi_plot(gradC_l, p=1, which='RMSprop', nIter=1000, alpha=1e-5, M=1, betas=0.99, log=False, label='grad H')
