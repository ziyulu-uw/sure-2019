# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: June 2019
# Description: This program finds the optimal Kalman gain using stochastic gradient descent. \
# Gradients in this program are obtained from forward computation.

import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

# constants
m = 1  # mass of the object
k = 0.5  # spring constant
gamma = 0.1  # friction coefficient
omega = k / m
mu = gamma / m
w = math.sqrt(omega - 0.25 * mu ** 2)
N = 4  # number of time steps in one simulation
dt = 0.05  # step size in one simulation
sigma = 0.1  # noise coefficient in SDE
Q = 0.1  # observation noise variance
X0 = np.array([[1.0], [0.0]])  # initial state (X = [x, v]^T)
C = np.array([1.0, 0.0], ndmin=2)  # observation matrix

lambda1 = complex(-0.5 * mu, w)
lambda2 = complex(-0.5 * mu, -w)

# construct matrix A
A11 = lambda2 * cmath.exp(lambda1 * dt) - lambda1 * cmath.exp(lambda2 * dt)
A12 = -cmath.exp(lambda1 * dt) + cmath.exp(lambda2 * dt)
A21 = lambda1 * lambda2 * (cmath.exp(lambda1 * dt) - cmath.exp(lambda2 * dt))
A22 = -lambda1 * cmath.exp(lambda1 * dt) + lambda2 * cmath.exp(lambda2 * dt)
A = np.array([[A11, A12], [A21, A22]])
A = np.divide(A, lambda2 - lambda1)  # A turns out to be real
A = A.real

# print(A)

# construct covariance matrix R
e1 = cmath.exp(2 * lambda1 * dt)
e2 = cmath.exp(2 * lambda2 * dt)
e3 = cmath.exp((lambda1 + lambda2) * dt)
R11 = 0.5 * (e1 - 1) / lambda1 + 0.5 * (e2 - 1) / lambda2 - 2 * (e3 - 1) / (lambda1 + lambda2)
R12 = 0.5 * (e1 - 1) + 0.5 * (e2 - 1) - (e3 - 1)
R21 = R12
R22 = lambda1 * 0.5 * (e1 - 1) + lambda2 * 0.5 * (e2 - 1) - 2 * lambda1 * lambda2 * (e3 - 1) / (lambda1 + lambda2)
R = np.array([[R11, R12], [R21, R22]])
R = np.multiply(R, (sigma / (lambda2 - lambda1)) ** 2)  # R turns out to be real
R = R.real

# print(R)

def compute_gradient(K, z):
    # K -- Kalman gain, z -- random seed (set when testing)
    # computes dF/dK, where F = 1/2N * \sum_{n=1}^N (X_hat_n-X_n)^2
    # returns dF/dK, F

    # np.random.seed(z)  # set random seed for testing purpose
    X = X0  # initial state
    X_hat = X0  # initial state estimate
    dX_hat = np.zeros((2, 2))  # initial gradient dX_hat/dK
    sum_gradients = np.zeros((1, 2))
    sum_errors = 0

    for n in range(N):
        Z_hat = C @ A @ X_hat  # predicted observation
        W = np.random.multivariate_normal([0, 0], R)  # gaussian system noise with mean 0 covariance R
        W = np.array(W, ndmin=2)
        W = np.transpose(W)
        X = A @ X + W  # state update
        # print("State", X)
        V = np.random.normal(0, Q)  # gaussian observation noise with mean 0 variance Q
        Z = C @ X + V  # observation
        # print("Obs", Z)
        X_hat = A @ X_hat + K * (Z - Z_hat)  # state estimate
        # print("Est", X_hat)
        diag = Z - Z_hat
        diag = np.array([1, 1]) * diag[0][0]
        dX_hat = A @ dX_hat + np.diag(diag) - K @ C @ A @ dX_hat
        sum_gradients += np.transpose(X_hat - X) @ dX_hat
        error = X_hat - X
        sum_errors += (error[0][0] ** 2 + error[1][0] ** 2)

    return sum_gradients / N, sum_errors / (2 * N)


def stochastic_gradient_descent(K, n, alpha, z):
    # K -- initial Kalman gain, n -- number of gradient steps,\
    # alpha -- learning rate, z -- random seed
    # performs gradient descent using dF/dK as gradient
    # returns K, a list of F at each gradient step, a list of dF/dK at each gradient step

    err_L = []
    grad_L = []
    for i in range(n):
        grad, err = compute_gradient(K, z)
        err_L.append(err)
        grad_L.append(grad)
        # print(grad)
        K = K - alpha * np.transpose(grad)
        # print(K)

    return K, err_L, grad_L


def Stochastic_gradient_descent(K, n, alpha, z_l):
    # z_l -- a list of random seeds
    # a wrapper function that calls stochastic_gradient_descent(K, n, alpha, z) for z in z_l
    # and plots F vs n

    print("Initialization: K11 is {}, K12 is {}".format(K[0][0], K[1][0]))
    K_avg = np.array([[0.0], [0.0]])
    err_avg = np.zeros(n)

    for z in z_l:
        K, err_L, grad_L = stochastic_gradient_descent(K, n, alpha, z)
        print("Seed {}: After {:d} iterations, K11 becomes {:.3f}, K12 becomes {:.3f}. The final loss is {:.3f}".\
              format(z, n, K[0][0], K[1][0], err_L[-1]))
        print("Gradient for each iteration", grad_L)
        K_avg += K
        err_avg += err_L

    print("Averaging over {} random seeds, K11 is {:.3f}, K12 is {:.3f}. The final loss is {:.3f}". \
          format(len(z_l), K_avg[0][0], K_avg[1][0], err_avg[-1]))

    x = [i for i in range(n)]
    plt.plot(x, err_avg)
    plt.title("Stochastic gradient descent with {} steps and step size {}".format(str(n), str(alpha)))
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("mean squared error of one simulation")
    plt.show()


K = np.array([[1.0], [1.0]])
Stochastic_gradient_descent(K, 1000, 0.1, [1,2,3])
