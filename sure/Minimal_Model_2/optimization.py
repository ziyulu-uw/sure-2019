# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program imports optimization methods from PyTorch for the filter and control optimization problem

import torch
import numpy as np
from Cost import cost_computation
from Simulation import path_generator
from FinteDiffApprox import FDA


def optimize(init_cond, param_list,vn_list, Gb, Ib, meas_time, t_list, tk_list, qk_list, meal_time, dt, h, which, alpha, momentum, beta, n):
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
        grad = FDA(init_cond, param_list,vn_list, Gb, Ib, meas_time, t_list, tk_list, qk_list, meal_time, dt, h)
        G, X, I, Ra = path_generator(init_cond, param_list, vn_list, Gb, Ib, meas_time, t_list, tk_list,qk_list, meal_time, dt, h)
        cost = cost_computation(G, vn_list)
        cost_l.append(cost)
        grad_l.append(grad)
        # print(grad)
        # print(param_tensor.grad)
        grad_tensor = torch.tensor(grad, requires_grad=False)
        param_tensor.grad = grad_tensor
        optimizer.step()

    param_list = param_tensor.detach().numpy()
    G, X, I, Ra = path_generator(init_cond, param_list, vn_list, Gb, Ib, meas_time, t_list, tk_list, qk_list, meal_time,dt, h)
    cost = cost_computation(G, vn_list)
    cost_l.append(cost)

    return param_list, cost_l, grad_l
