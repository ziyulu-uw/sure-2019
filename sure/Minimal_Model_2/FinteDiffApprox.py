# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Use finite difference approximation to find out the dJ/d theta

from Cost import cost_computation
from Simulation import path_generator
import numpy as np


def FDA(init_cond, param_list,vn_list, Gb, Ib, meas_time, t_list, tk_list, qk_list, meal_time, dt, h):
    """take a list of parameters a, and return a list of dJ/da
    dJ/da = [J(a+s)-J(a-s)] / 2s """

    partial_derivative = np.zeros(len(param_list))
    for i in range(len(param_list)):
        s = 0.000001
        a = param_list[i]*(1+s)             #shift a parameter s of its original value
        b = param_list[i]*(1-s)

        param_shift_f = param_list.copy()   #make a copy of original parameter list
        param_shift_f[i] = a                #assign the new list with a shifted parameter

        param_shift_b = param_list.copy()  # make a copy of original parameter list
        param_shift_b[i] = b               # assign the new list with a shifted parameter

        G_f, X_f, I_f, Ra_f = path_generator(init_cond, param_shift_f, vn_list, Gb, Ib, meas_time, t_list, tk_list, qk_list, meal_time, dt,h)
        G_b, X_b, I_b, Ra_b = path_generator(init_cond, param_shift_b, vn_list,  Gb, Ib,meas_time, t_list, tk_list, qk_list, meal_time, dt,h)

        cost_f = cost_computation(G_f, vn_list)  #J(a+s)
        cost_b = cost_computation(G_b, vn_list)  #J(a-s)
        dJ_da = (cost_f-cost_b)/(s*2)            # [J(a+s)-J(a-s)]/ 2s
        partial_derivative[i] = dJ_da
    return partial_derivative
