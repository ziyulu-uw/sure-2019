# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program imports optimization methods from PyTorch for the filter and control optimization problem \
# user can choose to use/not use learning rate scheduler, to use/not use mini-batch, \
# to do/not do comparison with theoretical result

import torch
import numpy as np
import noise_generation
import path_generation
import loss_computation
import backward_grad
from numpy import linalg as LA


def optimize(X0, A, C, B, G, K, N, S, R, d_X, d_Z, d_U, r, n, act, L, g, alpha, momentum, M, s, comp, which):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G -- initial control gain, K -- initial Kalman gain, \
    # N -- number of total time steps, S -- observation noise covariance matrix, R -- system noise covariance matrix, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control, \
    # r -- scaling factor, n -- number of total gradient steps, \
    # act -- set to True to activate learning rate scheduler, \
    # L -- list of milestones, g -- multiplicative factor of learning rate decay, \
    # alpha -- learning rate, momentum -- momentum factor (can set to 0), \
    # M -- mini-batch size, s -- random seed, \
    # comp -- an array containing the theoretical optimal K, G, and cost, \
    # which -- name of the optimization algorithm to use (SGD, Adam, or RMSprop)
    # optimizes K and G using SGD algorithms
    # returns optimized K and G, a list of F at each gradient step, a list of dF/dK, dF/dG at each gradient step
    # if comparing with theoretical results, \
    # also returns the difference between the current K, G and the optimal K, G at each gradient step

    K_tensor = torch.tensor(K, requires_grad = True)
    G_tensor = torch.tensor(G, requires_grad=True)

    if which == 'SGD':
        optimizer = torch.optim.SGD([K_tensor, G_tensor], lr=alpha, momentum=momentum)
    elif which == 'Adam':
        optimizer = torch.optim.Adam([K_tensor, G_tensor], lr=alpha)
    elif which == 'RMSprop':
        optimizer = torch.optim.RMSprop([K_tensor, G_tensor], lr=alpha)
    else:
        print("Invalid algorithm")
        return

    if act:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=L, gamma=g)
    if comp is not None:
        best_K = comp[0]
        best_G = comp[1]
        diff_K_l = []
        diff_G_l = []

    np.random.seed(s)
    F_l = []
    grad_K_l = []
    grad_G_l = []

    for i in range(n):

        optimizer.zero_grad()
        K = K_tensor.detach().numpy()
        G = G_tensor.detach().numpy()

        if comp is not None:
            diff_K = K - best_K
            diff_G = G - best_G
            diff_K_l.append(LA.norm(diff_K))
            diff_G_l.append(LA.norm(diff_G))

        if M == 1:
            W = noise_generation.system_noise_generator(d_X, N, R)
            V = noise_generation.observation_noise_generator(d_Z, N, S)
            X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
            F = loss_computation.compute_loss(X, U, N, r)
            grad_K, grad_G = backward_grad.compute_gradient(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X)
            F_l.append(F)

        else:
            W = noise_generation.vectorized_system_noise_generator(M, d_X, N, R)
            V = noise_generation.vectorized_observation_noise_generator(M, d_Z, N, S)
            X, Z, U, X_hat = path_generation.multi_paths_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
            F = loss_computation.compute_multi_loss(X, U, N, r)
            grad_K, grad_G = backward_grad.compute_multi_gradients(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X)
            grad_K = np.mean(grad_K, axis=0)
            grad_G = np.mean(grad_G, axis=0)
            F = np.mean(F, axis=0)
            F_l.append(F[0][0])

        grad_K_l.append(grad_K)
        grad_G_l.append(grad_G)
        grad_K_tensor = torch.tensor(grad_K, requires_grad=False)
        grad_G_tensor = torch.tensor(grad_G, requires_grad=False)
        K_tensor.grad = grad_K_tensor
        G_tensor.grad = grad_G_tensor
        optimizer.step()
        if act:
            scheduler.step()

    K = K_tensor.detach().numpy()
    G = G_tensor.detach().numpy()

    if comp is not None:
        return K, G, F_l, grad_K_l, grad_G_l, diff_K_l, diff_G_l
    else:
        return K, G, F_l, grad_K_l, grad_G_l
