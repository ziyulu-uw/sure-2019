# Author: Ziyu Lu, Xinyu Li
# Email: zl1546@nyu.edu
# Date: August 2019
# Description: This program imports optimization methods from PyTorch for the filter and control optimization problem

import torch
import numpy as np
from Cost import cost_computation
from Path_whole import generate_path_whole
from FDA import FDA_control, FDA_filter, FDA_param


def optimize_param(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params,
                   which, alpha, momentum, beta, n):
    # which -- which optimization algorithm to use (SGD, Adam, or RMSprop), alpha -- learning rate, \
    # momentum -- momentum factor (can set to 0), n -- total number gradient steps,
    # optimizes parameters using SGD algorithms
    # returns optimized parameters, a list of cost at each gradient step, a list of gradient at each gradient step

    param_list = np.array(param_list)
    param_tensor = torch.tensor(param_list, requires_grad=True)

    if which == 'SGD':
        optimizer = torch.optim.SGD([param_tensor], lr=alpha, momentum=momentum)
    elif which == 'Adam':
        optimizer = torch.optim.Adam([param_tensor], lr=alpha)
    elif which == 'RMSprop':
        optimizer = torch.optim.RMSprop([param_tensor], lr=alpha, alpha=beta)
    else:
        print("Invalid algorithm")
        return

    cost_l = []
    grad_l = []

    for i in range(n):
        optimizer.zero_grad()
        param_list = param_tensor.detach().numpy()
        G, X, I, Ra, Z = generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas,
                                             T, T_list, N, meal_params)
        state_variable = [G, X, I, Ra]
        grad, cost = FDA_param(state_variable, init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas,
                               T, T_list, N, meal_params)
        cost_l.append(cost)
        grad_l.append(grad)
        # print(grad)
        # print(param_tensor.grad)
        grad_tensor = torch.tensor(grad, requires_grad=False)
        param_tensor.grad = grad_tensor
        optimizer.step()

    param_list = param_tensor.detach().numpy()
    G, X, I, Ra, Z = generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T,
                                         T_list, N, meal_params)
    cost = cost_computation(G, X, I, Ra, Gb, Ib, control_gain)
    cost_l.append(cost)

    return param_list, cost_l, grad_l


def optimize_control(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N,
                     meal_params, which, alpha, momentum, beta, n):
    # which -- which optimization algorithm to use (SGD, Adam, or RMSprop), alpha -- learning rate, \
    # momentum -- momentum factor (can set to 0), n -- total number gradient steps,
    # optimizes parameters using SGD algorithms
    # returns optimized parameters, a list of cost at each gradient step, a list of gradient at each gradient step

    control_gain = np.array(control_gain)
    control_tensor = torch.tensor(control_gain, requires_grad=True)

    if which == 'SGD':
        optimizer = torch.optim.SGD([control_tensor], lr=alpha, momentum=momentum)
    elif which == 'Adam':
        optimizer = torch.optim.Adam([control_tensor], lr=alpha)
    elif which == 'RMSprop':
        optimizer = torch.optim.RMSprop([control_tensor], lr=alpha, alpha=beta)
    else:
        print("Invalid algorithm")
        return

    cost_l = []
    grad_l = []

    for i in range(n):
        optimizer.zero_grad()
        control_gain = control_tensor.detach().numpy()
        G, X, I, Ra, Z = generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas,
                                             T, T_list, N, meal_params)
        state_variable = [G, X, I, Ra]
        grad, cost = FDA_control(state_variable, init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib,
                                 N_meas, T, T_list, N, meal_params)
        cost_l.append(cost)
        grad_l.append(grad)
        # print(grad)
        # print(param_tensor.grad)
        grad_tensor = torch.tensor(grad, requires_grad=False)
        control_tensor.grad = grad_tensor
        optimizer.step()

    control_gain = control_tensor.detach().numpy()
    G, X, I, Ra, Z = generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T,
                                         T_list, N, meal_params)
    cost = cost_computation(G, X, I, Ra, Gb, Ib, control_gain)
    cost_l.append(cost)

    return control_gain, cost_l, grad_l


def optimize_filter(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params,
                    which, alpha, momentum, beta, n):
    # which -- which optimization algorithm to use (SGD, Adam, or RMSprop), alpha -- learning rate, \
    # momentum -- momentum factor (can set to 0), n -- total number gradient steps,
    # optimizes parameters using SGD algorithms
    # returns optimized parameters, a list of cost at each gradient step, a list of gradient at each gradient step

    Filter = np.array(Filter)
    Filter_tensor = torch.tensor(Filter, requires_grad=True)

    if which == 'SGD':
        optimizer = torch.optim.SGD([Filter_tensor], lr=alpha, momentum=momentum)
    elif which == 'Adam':
        optimizer = torch.optim.Adam([Filter_tensor], lr=alpha)
    elif which == 'RMSprop':
        optimizer = torch.optim.RMSprop([Filter_tensor], lr=alpha, alpha=beta)
    else:
        print("Invalid algorithm")
        return

    cost_l = []
    grad_l = []

    for i in range(n):
        optimizer.zero_grad()
        Filter = Filter_tensor.detach().numpy()
        G, X, I, Ra, Z = generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas,
                                             T, T_list, N, meal_params)
        state_variable = [G, X, I, Ra]
        grad, cost = FDA_filter(state_variable, init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib,
                                N_meas, T, T_list, N, meal_params)
        cost_l.append(cost)
        grad_l.append(grad)
        # print(grad)
        # print(param_tensor.grad)
        grad_tensor = torch.tensor(grad, requires_grad=False)
        Filter_tensor.grad = grad_tensor
        optimizer.step()

    Filter = Filter_tensor.detach().numpy()
    G, X, I, Ra, Z = generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T,
                                         T_list, N, meal_params)
    cost = cost_computation(G, X, I, Ra, Gb, Ib, control_gain)
    cost_l.append(cost)

    return Filter, cost_l, grad_l

