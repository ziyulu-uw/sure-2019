# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: August 2019
# Description: This program implements a wrapper function for the optimization algorithms

from optimization_all import optimize_filter_control
# from optimization_sep import optimize_control
import numpy as np


def optim_wrapper(init_cond, param_list, control_gain, Filter, Gb, Ib, N_meas, T, T_list, N,
            meal_params, lbda, which, alpha, momentum, beta1, beta2, M, n, fname):
    # a wrapper function that calls the function optimize_filter_control in optimization_all.py
    # and saves the data to file fname

    valid_algo = ['Adam', 'RMSprop', 'SGD']
    if which not in valid_algo:
        print("Invalid algorithm")
        return

    print("Optimizing using {} algorithm".format(which))
    print("---- Initialization ----")
    print("The filter parameters are: K1={}, K2={}, K3={}, K4={}".format(Filter[0], Filter[1], Filter[2], Filter[3]))
    print("The control parameters are: H1={}, H2={}, H3={}, H4={}".
          format(control_gain[0], control_gain[1], control_gain[2], control_gain[3]))

    np.random.seed(1)

    Filter, control_gain, cost_l, filter_l, control_l, gradF_l, gradC_l = optimize_filter_control(init_cond, param_list, control_gain, Filter, Gb, Ib, N_meas, T, T_list, N,
                            meal_params, lbda, which, alpha, momentum, beta1, beta2, M, n)
    # control_l, cost_l, grad_l = optimize_control(init_cond, param_list, control_gain, Filter, Gb, Ib, N_meas, T, T_list, N,
    #                  meal_params, which, alpha, momentum, beta1, beta2, n)  # for testing control
    print("Initial loss: {:10.2e}".format(cost_l[0]))
    print("---- After {} iterations ----".format(n))
    print("The filter parameters are: K1={}, K2={}, K3={}, K4={}".format(Filter[0], Filter[1], Filter[2], Filter[3]))
    print("The control parameters are: H1={}, H2={}, H3={}, H4={}".
          format(control_gain[0], control_gain[1], control_gain[2], control_gain[3]))
    print("Final loss: {:10.2e}".format(cost_l[-1]))

    np.savez(fname, Filter=Filter, control_gain=control_gain, cost_l=cost_l, filter_l=filter_l, control_l=control_l, gradF_l=gradF_l, gradC_l=gradC_l)

    return Filter, control_gain, cost_l, filter_l, control_l, gradF_l, gradC_l
    # return control_l, cost_l, grad_l  # for testing control
