# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program uses optimization methods in PyTorch for the Kalman filter optimization problem

import torch
import numpy as np
import noise_generation
import path_generation
import loss_gradient_computation


def SGD(X0, A, C, N, R, S, K, n, momentum, alpha, s):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # N -- number of total time steps, R -- system noise covariance matrix, S -- observation noise covariance matrix, \
    # K -- transpose of initial Kalman gain, n -- number of total gradient steps, \
    # momentum -- momentum factor (can set to 0), alpha -- learning rate, s -- random seed
    # optimizes K using SGD algorithm
    # returns K^T, a list of F at each gradient step, a list of dF/dK at each gradient step

    K_tensor = torch.tensor(K, requires_grad = True)
    optimizer = torch.optim.SGD([K_tensor], lr=alpha, momentum=momentum)

    np.random.seed(s)
    d_X = len(X0)  # dimension of state
    d_Z = len(C)  # dimension of observation
    F_l = []
    grad_l = []

    for i in range(n):

        optimizer.zero_grad()
        W = noise_generation.system_noise_generator(d_X, N, R)
        V = noise_generation.observation_noise_generator(d_Z, N, S)
        X, Z = path_generation.path_generator(X0, A, C, N, W, V)

        K = K_tensor.detach().numpy()
        X_hat = path_generation.filtered_path_generator(X0, A, C, K, Z, N)
        F = loss_gradient_computation.compute_loss(X, X_hat, N)
        grad = loss_gradient_computation.compute_gradient(A, C, N, K, X, Z, X_hat)

        F_l.append(F)
        grad_l.append(grad)
        grad_tensor = torch.tensor(grad, requires_grad = False)
        K_tensor.grad = grad_tensor
        # print("grad", K_tensor.grad)
        optimizer.step()
        # print("K", K_tensor)

    K = K_tensor.detach().numpy()

    return K, F_l, grad_l


def Adam(X0, A, C, N, R, S, K, n, alpha, s):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # N -- number of total time steps, R -- system noise covariance matrix, S -- observation noise covariance matrix, \
    # K -- transpose of initial Kalman gain, n -- number of total gradient steps, \
    # alpha -- learning rate, s -- random seed
    # optimizes K using Adam algorithm
    # returns K^T, a list of F at each gradient step, a list of dF/dK at each gradient step

    K_tensor = torch.tensor(K, requires_grad = True)
    optimizer = torch.optim.Adam([K_tensor], lr=alpha)

    np.random.seed(s)
    d_X = len(X0)  # dimension of state
    d_Z = len(C)  # dimension of observation
    F_l = []
    grad_l = []

    for i in range(n):

        optimizer.zero_grad()
        W = noise_generation.system_noise_generator(d_X, N, R)
        V = noise_generation.observation_noise_generator(d_Z, N, S)
        X, Z = path_generation.path_generator(X0, A, C, N, W, V)

        K = K_tensor.detach().numpy()
        X_hat = path_generation.filtered_path_generator(X0, A, C, K, Z, N)
        F = loss_gradient_computation.compute_loss(X, X_hat, N)
        grad = loss_gradient_computation.compute_gradient(A, C, N, K, X, Z, X_hat)

        F_l.append(F)
        grad_l.append(grad)
        grad_tensor = torch.tensor(grad, requires_grad = False)
        K_tensor.grad = grad_tensor
        # print("grad", K_tensor.grad)
        optimizer.step()
        # print("K", K_tensor)

    K = K_tensor.detach().numpy()

    return K, F_l, grad_l


def RMSprop(X0, A, C, N, R, S, K, n, alpha, s):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # N -- number of total time steps, R -- system noise covariance matrix, S -- observation noise covariance matrix, \
    # K -- transpose of initial Kalman gain, n -- number of total gradient steps, \
    # alpha -- learning rate, s -- random seed
    # optimizes K using RMSprop algorithm
    # returns K^T, a list of F at each gradient step, a list of dF/dK at each gradient step

    K_tensor = torch.tensor(K, requires_grad = True)
    optimizer = torch.optim.RMSprop([K_tensor], lr=alpha)

    np.random.seed(s)
    d_X = len(X0)  # dimension of state
    d_Z = len(C)  # dimension of observation
    F_l = []
    grad_l = []

    for i in range(n):

        optimizer.zero_grad()
        W = noise_generation.system_noise_generator(d_X, N, R)
        V = noise_generation.observation_noise_generator(d_Z, N, S)
        X, Z = path_generation.path_generator(X0, A, C, N, W, V)

        K = K_tensor.detach().numpy()
        X_hat = path_generation.filtered_path_generator(X0, A, C, K, Z, N)
        F = loss_gradient_computation.compute_loss(X, X_hat, N)
        grad = loss_gradient_computation.compute_gradient(A, C, N, K, X, Z, X_hat)

        F_l.append(F)
        grad_l.append(grad)
        grad_tensor = torch.tensor(grad, requires_grad = False)
        K_tensor.grad = grad_tensor
        # print("grad", K_tensor.grad)
        optimizer.step()
        # print("K", K_tensor)

    K = K_tensor.detach().numpy()

    return K, F_l, grad_l



