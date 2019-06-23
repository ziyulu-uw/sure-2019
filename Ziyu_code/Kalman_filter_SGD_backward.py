# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: June 2019
# Description: This program finds the optimal Kalman gain using stochastic gradient descent. \
# Gradients in this program are obtained from backward computation.

import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

# constants
m = 1   # mass of the object
k = 0.5  # spring constant
gamma = 0.1  # friction coefficient
omega = k/m
mu = gamma/m
w = math.sqrt(omega - 0.25*mu**2)
N = 4  # number of time steps in one simulation
dt = 0.05  # step size in one simulation
sigma = 0.1  # noise coefficient in SDE
Q = 0.5  # observation noise variance
X0 = np.array([[1.0],[0.0]])  # initial state (X = [x, v]^T)
C = np.array([1.0, 0.0], ndmin=2)  # observation matrix


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

# print(A)

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

# print(R)

def generate_path(K, z):
    # K -- Kalman gain, z -- random seed (set when testing)
    # generates a path from the initial state X0 with Kalman gain K
    # returns a list of states, a list of observations, and a list of state estimations from the path

    # np.random.seed(z)  # set random seed for testing purpose
    X = X0  # initial state
    X_hat = X0  # initial state estimate
    L_state = []  # a list that stores the states in a path
    L_obs = []  # a list that stores the observations in a path
    L_est = []  # a list that stores the state estimations in a path
    L_state.append(X)
    L_obs.append(0)
    L_est.append(X_hat)

    for n in range(N):

        Z_hat = C @ A @ X_hat  # predicted observation
        W = np.random.multivariate_normal([0,0], R)  # gaussian system noise with mean 0 covariance R
        W = np.array(W, ndmin=2)
        W = np.transpose(W)
        X = A @ X + W  # state update
        L_state.append(X)  # stores the new state in the state list
        V = np.random.normal(0, Q)  # gaussian observation noise with mean 0 variance Q
        Z = C @ X + V  # observation
        L_obs.append(Z)  # stores the new observation in the observation list
        X_hat = A @ X_hat + K * (Z - Z_hat)  # state estimate
        L_est.append(X_hat)  # stores the new estimation in the estimation list

    # print("States", L_state)
    # print("Obs", L_obs)
    # print("Est", L_est)

    return L_state, L_obs, L_est


def compute_gradient(K, L_state, L_obs, L_est):
    # L_state -- list of states from one path, L_obs -- list of observations from one path, \
    # L_est -- list of state estimations from one path
    # computes dF/dK, where F = 1/2N *\sum_{n=1}^N (X_hat_n-X_n)^2
    # returns dF/dK

    Q = 0  # start with Q_N = [0, 0]
    P = 2*np.transpose(L_est[N] - L_state[N])  # start with P_N = 2*(\hat{X_N} - X_N)^T
    for i in range(N-1, -1, -1):  # move backward
        diag = L_obs[i+1] - C @ A @ L_est[i]
        diag = np.array([1, 1]) * diag[0][0]
        Q = Q + P @ np.diag(diag)
        P = 2*np.transpose(L_est[i] - L_state[i]) + P @ (np.identity(2) - K @ C) @ A

    return Q/(2*N)


def compute_error(L_state, L_est):
    # L_state -- list of states from one path, L_est -- list of state estimations from one path
    # computes the mean-squared error: F = 1/2N *\sum_{n=1}^N (X_hat_n-X_n)^2
    # returns F

    F = 0
    assert (N == len(L_state)-1), "Number of intended time steps and number of states not equal. Something is wrong."
    for i in range(1, len(L_state)):
        error = L_state[i] - L_est[i]
        F += (error[0][0]**2 + error[1][0]**2)
    return F/(2*N)

def stochastic_gradient_descent(K, n, alpha, z):
    # K -- initial Kalman gain, n -- number of gradient steps, \
    # alpha -- learning rate, z -- random seed
    # performs gradient descent using dF/dK as gradient
    # returns K, a list of F at each gradient step, a list of dF/dK at each gradient step

    np.random.seed(z)  # set random seed
    err_L = []
    grad_L = []
    for i in range(n):
        L_state, L_obs, L_est = generate_path(K, z)
        grad = compute_gradient(K, L_state, L_obs, L_est)
        err = compute_error(L_state, L_est)
        err_L.append(err)
        grad_L.append(grad)
        # print(grad)
        K = K - alpha * np.transpose(grad)
        # print(K)

    return K, err_L, grad_L

def Stochastic_gradient_descent(K0, n,alpha, z_l):
    # z_l -- a list of random seeds
    # a wrapper function that calls stochastic_gradient_descent(K, n, alpha, z) for z in z_l \
    # and plots F vs n

    print("Initialization: K11 is {}, K12 is {}".format(K0[0][0], K0[1][0]))
    K_avg = np.array([[0.0], [0.0]])
    err_avg = np.zeros(n)

    # plots F vs n
    for z in z_l:
        K, err_L, grad_L = stochastic_gradient_descent(K0, n, alpha, z)
        print("Seed {}: After {:d} iterations, K11 is {:.3f}, K12 is {:.3f}. The final loss is {:.3f}".\
              format(z, n, K[0][0], K[1][0], err_L[-1]))
        print("Gradient for each iteration", grad_L)
        K_avg += K
        err_avg += err_L

    K_avg = K_avg/len(z_l)
    err_avg = err_avg/len(z_l)

    print("Averaging over {} random seeds, K11 is {:.3f}, K12 is {:.3f}. The final loss is {:.3f}".\
          format(len(z_l), K_avg[0][0], K_avg[1][0], err_avg[-1]))

    x = [i for i in range(n)]
    plt.plot(x, err_avg)
    plt.title("Stochastic gradient descent with {} steps and step size {}, averaged over {} random seeds".\
              format(str(n),str(alpha),str(len(z_l))))
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("mean squared error of one simulation")
    plt.show()

    return K_avg, err_avg

######## Testing code ###########
def finite_diff_approx(K, delta_K, z):
    # K -- Kalman gain, delta_K -- difference in finite difference approximation, z -- random seed
    # performs first order finite difference approximation of dF/dK: dF/dK = (F(K+delta_K) - F(K))/delta_K
    # returns the finite difference approximation of dF/dK, and F

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
        V = np.random.normal(0, Q)  # gaussian observation noise with mean 0 variance Q
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
            V = np.random.normal(0, Q)  # gaussian observation noise with mean 0 variance Q
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
    L_state, L_obs, L_est = generate_path(K, 1)
    grad = compute_gradient(L_state, L_obs, L_est)
    # print(np.linalg.norm(grad))
    print(grad)  # This is the gradient computed by formulas
    for i in range(n):
        delta_K_n = delta_K * (i+1)
        grad_approx, F = finite_diff_approx(K, delta_K_n, z)
        print(grad_approx)  # This is the gradient given by finite difference approximation
        x_L.append(delta_K_n)
        y_L.append(np.linalg.norm(grad_approx - grad))

    norm1 = np.linalg.norm(grad)
    norm2 = y_L[0]
    print(norm2/norm1)
    plt.plot(x_L, y_L)
    plt.title("First order finite difference approximation of dF/dK")
    plt.xlabel("delta_K")
    plt.ylabel("Frobenius norm of (grad_approx - grad)")
    plt.show()


K = np.array([[5.0], [5.0]])

# check_order(K, 0.0001, 1, 10)

K_, err_L = Stochastic_gradient_descent(K, 10000, 0.0001, [1, 5, 10, 20, 27])
# print("------Train history------")
# print("Loss for each iteration:", err_L)


