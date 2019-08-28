# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Use finite difference approximation to find out the dJ/d theta

from Cost import cost_computation
from Path_whole import generate_path_whole
import numpy as np
"""a more efficient version"""

def FDA_param(state_variable, init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params):
    """take a list of parameters a, and return a list of dJ/da
    dJ/da = [J(a+s)-J(a-s)] / 2s
    @:return 3 arrays: partial derivatives of parameter, filter, and control"""
    G, X, I, Ra = state_variable
    param_partial_derivative = np.zeros(len(param_list))
    for i in range(len(param_list)):
        s = 0.000005
        a = param_list[i]*(1+s)             #shift a parameter s of its original value
        b = param_list[i]

        param_shift_f = param_list.copy()   #make a copy of original parameter list
        param_shift_f[i] = a                #assign the new list with a shifted parameter

        G_f, X_f, I_f, Ra_f, mf = generate_path_whole(init_cond, param_shift_f, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)

        cost_f = cost_computation(G_f, X_f, I_f, Ra_f, Gb, Ib, control_gain) #J(a+s)

        cost = cost_computation(G, X, I, Ra, Gb, Ib, control_gain) #J(a-s)

        dJ_da = (cost_f-cost)/(a-b)            # [J(a)-J(b)]/ 2(a-b)
        param_partial_derivative[i] = dJ_da
    return param_partial_derivative, cost

def FDA_control(state_variable, init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params):
    G, X, I, Ra = state_variable
    control_partial_derivative = np.zeros(len(control_gain))
    for i in range(len(control_gain)):
        s = 0.0005
        a = control_gain[i] * (1 + s)  # shift a parameter s of its original value
        b = control_gain[i]

        control_shift_f = control_gain.copy()  # make a copy of original parameter list
        control_shift_f[i] = a  # assign the new list with a shifted parameter

        G_f, X_f, I_f, Ra_f, mf = generate_path_whole(init_cond, param_list, control_shift_f, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)
        cost_f = cost_computation(G_f, X_f, I_f, Ra_f, Gb, Ib, control_shift_f)  # J(a+s)
        cost = cost_computation(G, X, I, Ra, Gb, Ib, control_gain)  # J(a-s)
        dJ_da = (cost_f - cost) / (a-b)   # [J(a)-J(b)]/ 2(a-b)
        control_partial_derivative[i] = dJ_da
    return control_partial_derivative, cost


def FDA_filter(state_variable, init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params):
    G, X, I, Ra = state_variable

    filter_partial_derivative = np.zeros(len(Filter))
    for i in range(len(Filter)):
        s = 0.005
        a = Filter[i] * (1 + s)  # shift a parameter s of its original value
        b = Filter[i]

        filter_shift_f = Filter.copy()  # make a copy of original parameter list
        filter_shift_f[i] = a  # assign the new list with a shifted parameter

        G_f, X_f, I_f, Ra_f, mf = generate_path_whole(init_cond, param_list, control_gain, filter_shift_f, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)
        cost_f = cost_computation(G_f, X_f, I_f, Ra_f, Gb, Ib, control_gain)  # J(a+s)
        cost = cost_computation(G, X, I, Ra, Gb, Ib, control_gain)  # J(a-s)
        dJ_da = (cost_f - cost) / (a-b)   # [J(a)-J(b)]/ 2(a-b)
        filter_partial_derivative[i] = dJ_da
    return filter_partial_derivative, cost