# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: June 2019
# Description: This program finds the optimal Kalman gain using gradient descent and stochastic gradient descent \
# This program also includes testing code for gradient using finite difference approximation.


import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

# constants
m = 1   # mass of the object
k = 0.5  # spring constant
gamma = 0.01  # friction coefficient
omega = k/m
mu = gamma/m
w = math.sqrt(omega - 0.25*mu**2)
N = 100  # number of time steps in one simulation
dt = 0.05  # step size in one simulation
sigma = 0.1  # noise coefficient in SDE
var_v = 0.05  # observation noise variance
X0 = np.array([[1.0],[0.0]])  # initial state (X = [x, v]^T)
C = np.array([1.0, 0.0], ndmin=2) # observation matrix


lambda1 = complex(-0.5*mu,w)
lambda2 = complex(-0.5*mu,-w)

# construct matrix A
A11 = lambda2*cmath.exp(lambda1*dt) - lambda1*cmath.exp(lambda2*dt)
A12 = -cmath.exp(lambda1*dt) + cmath.exp(lambda2*dt)
A21 = lambda1*lambda2*(cmath.exp(lambda1*dt) - cmath.exp(lambda2*dt))
A22 = -lambda1*cmath.exp(lambda1*dt) + lambda2*cmath.exp(lambda2*dt)
A = np.array([[A11, A12], [A21, A22]])
A = np.divide(A, lambda2-lambda1)  # A turns out to be real
A = A.real

# construct covariance matrix R
e1 = cmath.exp(2*lambda1*dt)
e2 = cmath.exp(2*lambda2*dt)
e3 = cmath.exp((lambda1 + lambda2)*dt)
R11 = 0.5*(e1-1)/lambda1 + 0.5*(e2-1)/lambda2 - 2*(e3-1)/(lambda1 + lambda2)
R12 = 0.5*(e1-1) + 0.5*(e2-1) - (e3-1)
R21 = R12
R22 = lambda1*0.5*(e1-1) + lambda2*0.5*(e2-1) - 2*lambda1*lambda2*(e3-1)/(lambda1 + lambda2)
R = np.array([[R11, R12], [R21, R22]])
R = np.multiply(R, (sigma/(lambda2-lambda1))**2)  # R turns out to be real
R = R.real


def compute_gradient(K, z):
    # K -- Kalman gain, z -- random seed
    # computes dF/dK, where F = 1/2N * \sum_{n=1}^N (X_hat_n-X_n)^2
    # returns dF/dK, F

    np.random.seed(z)  # set random seed
    X = X0  # initial state
    X_hat = X0  # initial state estimate
    dX_hat = np.zeros((2,2))  # initial gradient dX_hat/dK
    sum_gradients = np.zeros((1,2))
    sum_errors = 0

    for n in range(N):

        Z_hat = C @ A @ X_hat  # predicted observation
        W = np.random.multivariate_normal([0,0], R)  # gaussian system noise with mean 0 covariance R
        W = np.array(W, ndmin=2)
        W = np.transpose(W)
        X = A @ X + W  # state update
        V = np.random.normal(0, var_v)  # gaussian observation noise with mean 0 variance var_v
        Z = C @ X + V  # observation
        X_hat = A @ X_hat + K * (Z - Z_hat)  # state estimate
        diag = Z - Z_hat
        diag = np.array([1,1]) * diag[0][0]
        dX_hat = A @ dX_hat + np.diag(diag) - K @ C @ A @ dX_hat
        sum_gradients += np.transpose(X_hat - X) @ dX_hat
        error = X_hat - X
        sum_errors += (error[0][0]**2 + error[1][0]**2)

    return sum_gradients/N, sum_errors/(2*N)


def stochastic_gradient_descent(n, alpha, z):
    # n -- number of gradient steps, alpha -- step size, z -- random seed
    # performs gradient descent using dF/dK as gradient
    # returns K, a list of F at each gradient step, a list of dF/dK at each gradient step

    # delta_K = 0.0001
    K = np.array([[-1], [-1]])  # initialize K
    err_L = []
    grad_L = []
    for i in range(n):
        grad, err = compute_gradient(K, z)
        # grad, err = finite_diff_approx(K, delta_K, z)
        err_L.append(err)
        grad_L.append(grad)
        K = K - alpha*np.transpose(grad)
        # print(K)

    return K, err_L, grad_L


def gradient_descent(n, alpha, m, z):
    # n -- number of gradient steps, alpha -- step size, m -- number of simulations to compute expectation, \
    # z -- random seed
    # performs gradient descent using E[dF/dK] as gradient
    # returns K, a list of E[F] at each gradient step, a list of E[dF/dK] at each gradient step

    K = np.array([[1.0], [1.0]])  # initialize K
    err_L = []
    grad_L = []
    for i in range(n):
        sum_grad = np.zeros((1,2))
        sum_err = 0
        for j in range(m):
            grad, err = compute_gradient(K, z)
            sum_grad += grad
            sum_err += err
        exp_grad = sum_grad/m  # E[dF/dK] is approximated by 1/m * \sum_{i=1}^N dF_i/dK
        exp_err = sum_err/m    # E[F] is approximated by 1/m * \sum_{i=1}^N F_i
        err_L.append(exp_err)
        grad_L.append(exp_grad)
        K = K - alpha * np.transpose(exp_grad)
        # print(K)

    return K, err_L, grad_L


def Stochastic_gradient_descent(n,alpha, z):
    # calls stochastic_gradient_descent(n, alpha, z) and plots F vs n, log(dF/dK) vs n

    K, err_L, grad_L = stochastic_gradient_descent(n, alpha, z)
    print(K)
    print(err_L)
    x = [i for i in range(n)]
    plt.plot(x, err_L)
    plt.title("Stochastic gradient descent with {} steps and step size {}".format(str(n),str(alpha)))
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("mean squared error of one simulation")
    plt.show()

    grad1_L = [grad_L[i][0][0] for i in range(n)]
    grad2_L = [grad_L[i][0][1] for i in range(n)]
    log_grad1_L = [math.log10(abs(grad1_L[i])) for i in range(n)]
    log_grad2_L = [math.log10(abs(grad2_L[i])) for i in range(n)]
    plt.plot(x, log_grad1_L, label='grad1')
    plt.plot(x, log_grad2_L, label='grad2')
    plt.title("Stochastic gradient descent with {} steps and step size {}".format(str(n), str(alpha)))
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("log10(gradient)")
    plt.legend()
    plt.show()



def Gradient_descent(n, alpha, m, z):
    # calls gradient_descent(n, alpha, m, z) and plots E[F] vs n, log(E[dF/dK]) vs n

    K, err_L, grad_L = gradient_descent(n, alpha, m, z)
    print(K)
    print(err_L)
    x = [i for i in range(n)]
    plt.plot(x, err_L)
    plt.title("Expected gradient descent with {} steps and step size {}".format(str(n), str(alpha)))
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("expected mean squared error of one simulation")
    plt.show()

    grad1_L = [grad_L[i][0][0] for i in range(n)]
    grad2_L = [grad_L[i][0][1] for i in range(n)]
    log_grad1_L = [math.log10(abs(grad1_L[i])) for i in range(n)]
    log_grad2_L = [math.log10(abs(grad2_L[i])) for i in range(n)]
    plt.plot(x, log_grad1_L, label='grad1')
    plt.plot(x, log_grad2_L, label='grad2')
    plt.title("Expected gradient descent with {} steps and step size {}".format(str(n), str(alpha)))
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("log10(gradient)")
    plt.legend()
    plt.show()


# Testing code
def finite_diff_approx(K, delta_K, z):
    # K -- Kalman gain, delta_K -- difference in finite difference approximation, z -- random seed
    # performs first order finite difference approximation of dF/dK: dF/dK = (F(K+delta_K) - F(K))/delta_K
    # returns dF/dK, and a finite difference approximation of dF/dK

    grad_approx = np.array([[0.0, 0.0]])

    # compute F(K)
    np.random.seed(z)  # set random seed
    X = X0  # initial state
    X_hat = X0  # initial state estimate
    sum_errors = 0

    for n in range(N):
        Z_hat = C @ A @ X_hat  # predicted observation
        W = np.random.multivariate_normal([0, 0], R)  # gaussian system noise with mean 0 covariance R
        W = np.array(W, ndmin=2)
        W = np.transpose(W)
        X = A @ X + W  # state update
        V = np.random.normal(0, var_v)  # gaussian observation noise with mean 0 variance var_v
        Z = C @ X + V  # observation
        X_hat = A @ X_hat + K * (Z - Z_hat)  # state estimate
        error = X_hat - X
        sum_errors += error[0][0] ** 2 + error[1][0] ** 2

    F = sum_errors/(2*N)

    # compute F(K+delta_K)
    for i in range(len(K)):

        np.random.seed(z)  # set random seed
        X = X0  # initial state
        X_hat = X0  # initial state estimate
        K[i][0] += delta_K
        sum_errors = 0

        for n in range(N):
            Z_hat = C @ A @ X_hat  # predicted observation
            W = np.random.multivariate_normal([0, 0], R)  # gaussian system noise with mean 0 covariance R
            W = np.array(W, ndmin=2)
            W = np.transpose(W)
            X = A @ X + W  # state update
            V = np.random.normal(0, var_v)  # gaussian observation noise with mean 0 variance var_v
            Z = C @ X + V  # observation
            X_hat = A @ X_hat + K * (Z - Z_hat)  # state estimate
            error = X_hat - X
            sum_errors += error[0][0] ** 2 + error[1][0] ** 2

        F_ = sum_errors/(2*N)
        grad_approx[0][i] = (F_ - F)/delta_K
        K[i][0] -= delta_K

    return grad_approx, F


def check_order(K, delta_K, z, n):
    # K -- Kalman gain, delta_K -- difference in finite difference approximation, \
    # z -- random seed, n -- largest multiple of delta_K
    # checks if the above first order approximation works properly \
    # by plotting the norm of approximation error against the difference
    # the plot should be linear

    x_L = []
    y_L = []
    grad, err = compute_gradient(K, z)
    print(grad)
    for i in range(n):
        delta_K_n = delta_K * (i+1)
        grad_approx, F = finite_diff_approx(K, delta_K_n, z)
        x_L.append(delta_K_n)
        y_L.append(np.linalg.norm(grad_approx - grad))

    # print(y_L)
    plt.plot(x_L, y_L)
    plt.show()


# Testing
# K = np.array([[2.0], [2.0]])
# check_order(K, 0.00001,1,5)

Stochastic_gradient_descent(500, 1e-5, 1)
# Gradient_descent(500, 0.001, 10, 1)



