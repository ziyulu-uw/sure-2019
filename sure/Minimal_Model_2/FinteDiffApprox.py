# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Use finite difference approximation to find out the dJ/d theta

from Cost import cost_computation
from Path_whole import generate_path_whole
import numpy as np


def FDA_param(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params):
    """take a list of parameters a, and return a list of dJ/da
    dJ/da = [J(a+s)-J(a-s)] / 2s
    @:return 3 arrays: partial derivatives of parameter, filter, and control"""

    param_partial_derivative = np.zeros(len(param_list))
    for i in range(len(param_list)):
        s = 0.0000001
        a = param_list[i]*(1+s)             #shift a parameter s of its original value
        b = param_list[i]*(1-s)

        param_shift_f = param_list.copy()   #make a copy of original parameter list
        param_shift_f[i] = a                #assign the new list with a shifted parameter

        param_shift_b = param_list.copy()  # make a copy of original parameter list
        param_shift_b[i] = b               # assign the new list with a shifted parameter

        G_f, X_f, I_f, Ra_f, mf = generate_path_whole(init_cond, param_shift_f, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)
        G_b, X_b, I_b, Ra_b, mb = generate_path_whole(init_cond, param_shift_b, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)
        cost_f = cost_computation(G_f, X_f, I_f, Ra_f, Gb, Ib, control_gain)#J(a+s)
        cost_b = cost_computation(G_b, X_b, I_b, Ra_b, Gb, Ib, control_gain) #J(a-s)
        dJ_da = (cost_f-cost_b)/(s * 2)            # [J(a+s)-J(a-s)]/ 2s
        param_partial_derivative[i] = dJ_da
    return param_partial_derivative

def FDA_control(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params):
    control_partial_derivative = np.zeros(len(control_gain))
    for i in range(len(control_gain)):
        s = 0.0000001
        a = control_gain[i] * (1 + s)  # shift a parameter s of its original value
        b = control_gain[i] * (1 - s)

        control_shift_f = control_gain.copy()  # make a copy of original parameter list
        control_shift_f[i] = a  # assign the new list with a shifted parameter

        control_shift_b = control_gain.copy()  # make a copy of original parameter list
        control_shift_b[i] = b  # assign the new list with a shifted parameter

        G_f, X_f, I_f, Ra_f, mf = generate_path_whole(init_cond, param_list, control_shift_f, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)
        G_b, X_b, I_b, Ra_b, mb = generate_path_whole(init_cond, param_list, control_shift_b, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)
        cost_f = cost_computation(G_f, X_f, I_f, Ra_f, Gb, Ib, control_gain)  # J(a+s)
        cost_b = cost_computation(G_b, X_b, I_b, Ra_b, Gb, Ib, control_gain)  # J(a-s)
        dJ_da = (cost_f - cost_b) / (s * 2)  # [J(a+s)-J(a-s)]/ 2s
        control_partial_derivative[i] = dJ_da
    return control_partial_derivative


def FDA_filter(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params):
    filter_partial_derivative = np.zeros(len(Filter))
    for i in range(len(Filter)):
        s = 0.0000001
        a = Filter[i] * (1 + s)  # shift a parameter s of its original value
        b = Filter[i] * (1 - s)

        filter_shift_f = Filter.copy()  # make a copy of original parameter list
        filter_shift_f[i] = a  # assign the new list with a shifted parameter

        filter_shift_b = Filter.copy()  # make a copy of original parameter list
        filter_shift_b[i] = b  # assign the new list with a shifted parameter

        G_f, X_f, I_f, Ra_f, mf = generate_path_whole(init_cond, param_list, control_gain, filter_shift_f, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)
        G_b, X_b, I_b, Ra_b, mb = generate_path_whole(init_cond, param_list, control_gain, filter_shift_b, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)
        cost_f = cost_computation(G_f, X_f, I_f, Ra_f, Gb, Ib, control_gain)  # J(a+s)
        cost_b = cost_computation(G_b, X_b, I_b, Ra_b, Gb, Ib, control_gain)  # J(a-s)
        dJ_da = (cost_f - cost_b) / (s * 2)  # [J(a+s)-J(a-s)]/ 2s
        filter_partial_derivative[i] = dJ_da
    return filter_partial_derivative