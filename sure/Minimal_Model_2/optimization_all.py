# Author: Ziyu Lu, Xinyu Li
# Email: zl1546@nyu.edu
# Date: August 2019
# Description: This program imports optimization methods from PyTorch for the filter and control optimization problem

import torch
import numpy as np
from Cost import cost_computation
from Noise_generator import noise_path
from Path_whole import generate_path_whole
from FDA import FDA_control, FDA_filter, FDA_param


def optimize_filter_control(init_cond, param_list, control_gain, Filter, Gb, Ib, N_meas, T, T_list, N,
                            meal_params, which, alpha, momentum, beta1, beta2, M, n):
    # which -- which optimization algorithm to use (SGD, Adam, or RMSprop), \
    # alpha -- learning rate, \
    # momentum -- momentum factor (can set to 0), \
    # betas -- coefficients for computing running average (set to None to use the default values), \
    # M -- mini-batch size, \
    # n -- total number gradient steps, \
    # optimizes filter and control using SGD algorithms
    # returns optimized parameters, a list of cost at each gradient step, a list of gradient at each gradient step

    FC = np.concatenate((np.array(Filter), np.array(control_gain)), axis=0)
    FC_tensor = torch.tensor(FC, requires_grad=True)

    if which == 'SGD':
        optimizer = torch.optim.SGD([FC_tensor], lr=alpha, momentum=momentum)
    elif which == 'Adam':
        optimizer = torch.optim.Adam([FC_tensor], lr=alpha, betas=(beta1,beta2))
    elif which == 'RMSprop':
        optimizer = torch.optim.RMSprop([FC_tensor], lr=alpha, alpha=beta1)
    else:
        print("Invalid algorithm")
        return

    cost_l = []
    filter_l = np.zeros((4, n+1))
    control_l = np.zeros((4, n+1))
    gradF_l = np.zeros((4, n))
    gradC_l = np.zeros((4, n))

    for i in range(n):
        optimizer.zero_grad()
        FC = FC_tensor.detach().numpy()
        Filter = FC[:len(Filter)]
        control_gain = FC[len(Filter):]
        total_noise = noise_path(init_cond, N * N_meas)
        state_variable, Z, true_state_variable = generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas,
                                             T, T_list, N, meal_params)
        #true_G = true_state_variable[0]
        gradF, cost = FDA_filter(state_variable, true_state_variable,  init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib,
                                 N_meas, T, T_list, N, meal_params)
        gradC, cost = FDA_control(state_variable, true_state_variable, init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib,
                                 N_meas, T, T_list, N, meal_params)
        grad = np.concatenate((np.array(gradF), np.array(gradC)), axis=0)
        cost_l.append(cost)
        filter_l[:,i] = Filter
        control_l[:,i] = control_gain
        gradF_l[:,i] = np.array(gradF)
        gradC_l[:,i] = np.array(gradC)
        grad_tensor = torch.tensor(grad, requires_grad=False)
        FC_tensor.grad = grad_tensor
        optimizer.step()

    FC = FC_tensor.detach().numpy()
    Filter = FC[:len(Filter)]
    control_gain = FC[len(Filter):]
    total_noise = noise_path(init_cond, N * N_meas)
    state_variable, Z, true_state_variable = generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T,
                                         T_list, N, meal_params)
    cost = cost_computation(true_state_variable, state_variable, Gb, Ib, control_gain)
    cost_l.append(cost)
    filter_l[:, n] = Filter
    control_l[:, n] = control_gain

    return Filter, control_gain, cost_l, filter_l, control_l, gradF_l, gradC_l
