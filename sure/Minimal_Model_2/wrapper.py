# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: August 2019
# Description: This program implements a wrapper function for the optimization algorithms

from optimization_all import optimize_filter_control
import numpy as np
import matplotlib.pyplot as plt

def wrapper(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N,
                            meal_params, which, alpha, momentum, betas, n):

