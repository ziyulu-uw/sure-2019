# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program uses optimization methods in PyTorch for the filter and control optimization problem

import torch
import numpy as np
import noise_generation
import path_generation
import loss_computation
import backward_grad


def SGD(X0, A, C, B, G, K, N, S, R, d_X, d_Z, d_U, r, n, L, g, momentum, alpha, s):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G -- initial control gain, K -- initial Kalman gain, \
    # N -- number of total time steps, S -- observation noise covariance matrix, R -- system noise covariance matrix, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control, \
    # r -- scaling factor, n -- number of total gradient steps, \
    # L -- list of milestones, g -- multiplicative factor of learning rate decay, \
    # momentum -- momentum factor (can set to 0), alpha -- learning rate, s -- random seed
    # optimizes K and G using SGD algorithm
    # returns optimized K and G, a list of F at each gradient step, a list of dF/dK, dF/dG at each gradient step

    K_tensor = torch.tensor(K, requires_grad=True)
    G_tensor = torch.tensor(G, requires_grad=True)
    optimizer = torch.optim.SGD([K_tensor, G_tensor], lr=alpha, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=L, gamma=g)

    np.random.seed(s)
    F_l = []
    grad_K_l = []
    grad_G_l = []

    for i in range(n):

        optimizer.zero_grad()
        K = K_tensor.detach().numpy()
        G = G_tensor.detach().numpy()
        W = noise_generation.system_noise_generator(d_X, N, R)
        V = noise_generation.observation_noise_generator(d_Z, N, S)
        X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
        F = loss_computation.compute_loss(X, U, N, r)
        grad_K, grad_G = backward_grad.compute_gradient(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X)

        F_l.append(F)
        grad_K_l.append(grad_K)
        grad_G_l.append(grad_G)
        grad_K_tensor = torch.tensor(grad_K, requires_grad=False)
        grad_G_tensor = torch.tensor(grad_G, requires_grad=False)
        K_tensor.grad = grad_K_tensor
        G_tensor.grad = grad_G_tensor
        optimizer.step()
        scheduler.step()

    K = K_tensor.detach().numpy()
    G = G_tensor.detach().numpy()

    return K, G, F_l, grad_K_l, grad_G_l


def Adam(X0, A, C, B, G, K, N, S, R, d_X, d_Z, d_U, r, n, L, g, alpha, s):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G -- initial control gain, K -- initial Kalman gain, \
    # N -- number of total time steps, S -- observation noise covariance matrix, R -- system noise covariance matrix, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control, \
    # r -- scaling factor, n -- number of total gradient steps, \
    # L -- list of milestones, g -- multiplicative factor of learning rate decay, \
    # alpha -- learning rate, s -- random seed
    # optimizes K and G using Adam algorithm
    # returns optimized K and G, a list of F at each gradient step, a list of dF/dK, dF/dG at each gradient step

    K_tensor = torch.tensor(K, requires_grad = True)
    G_tensor = torch.tensor(G, requires_grad=True)
    optimizer = torch.optim.Adam([K_tensor, G_tensor], lr=alpha)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=L, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=L, gamma=g)

    np.random.seed(s)
    F_l = []
    grad_K_l = []
    grad_G_l = []

    for i in range(n):

        optimizer.zero_grad()
        K = K_tensor.detach().numpy()
        G = G_tensor.detach().numpy()
        W = noise_generation.system_noise_generator(d_X, N, R)
        V = noise_generation.observation_noise_generator(d_Z, N, S)
        X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
        F = loss_computation.compute_loss(X, U, N, r)
        grad_K, grad_G = backward_grad.compute_gradient(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X)

        F_l.append(F)
        grad_K_l.append(grad_K)
        grad_G_l.append(grad_G)
        grad_K_tensor = torch.tensor(grad_K, requires_grad=False)
        grad_G_tensor = torch.tensor(grad_G, requires_grad=False)
        K_tensor.grad = grad_K_tensor
        G_tensor.grad = grad_G_tensor
        optimizer.step()
        scheduler.step()

    K = K_tensor.detach().numpy()
    G = G_tensor.detach().numpy()

    return K, G, F_l, grad_K_l, grad_G_l


def RMSprop(X0, A, C, B, G, K, N, S, R, d_X, d_Z, d_U, r, n, L, g, alpha, s):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G -- initial control gain, K -- initial Kalman gain, \
    # N -- number of total time steps, S -- observation noise covariance matrix, R -- system noise covariance matrix, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control, \
    # r -- scaling factor, n -- number of total gradient steps, \
    # L -- list of milestones, g -- multiplicative factor of learning rate decay, \
    # alpha -- learning rate, s -- random seed
    # optimizes K and G using RMSprop algorithm
    # returns optimized K and G, a list of F at each gradient step, a list of dF/dK, dF/dG at each gradient step

    K_tensor = torch.tensor(K, requires_grad = True)
    G_tensor = torch.tensor(G, requires_grad=True)
    optimizer = torch.optim.Adam([K_tensor, G_tensor], lr=alpha)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=L, gamma=g)

    np.random.seed(s)
    F_l = []
    grad_K_l = []
    grad_G_l = []

    for i in range(n):

        optimizer.zero_grad()
        K = K_tensor.detach().numpy()
        G = G_tensor.detach().numpy()
        W = noise_generation.system_noise_generator(d_X, N, R)
        V = noise_generation.observation_noise_generator(d_Z, N, S)
        X, Z, U, X_hat = path_generation.path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
        F = loss_computation.compute_loss(X, U, N, r)
        grad_K, grad_G = backward_grad.compute_gradient(A, C, B, G, K, N, X, Z, U, X_hat, r, d_X)

        F_l.append(F)
        grad_K_l.append(grad_K)
        grad_G_l.append(grad_G)
        grad_K_tensor = torch.tensor(grad_K, requires_grad=False)
        grad_G_tensor = torch.tensor(grad_G, requires_grad=False)
        K_tensor.grad = grad_K_tensor
        G_tensor.grad = grad_G_tensor
        optimizer.step()
        scheduler.step()

    K = K_tensor.detach().numpy()
    G = G_tensor.detach().numpy()

    return K, G, F_l, grad_K_l, grad_G_l
