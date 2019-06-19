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
N = 100  # number of time steps in one simulation
dt = 0.05  # step size in one simulation
sigma = 0.1  # noise coefficient in SDE
var_v = 0.1  # observation noise variance
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

def generate_path(K, z):
    # K -- Kalman gain, z -- random seed
    # generates a path from the initial state X0 with Kalman gain K
    # returns a list of states, a list of observations, and a list of state estimations from the path

    np.random.seed(z)  # set random seed
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
        V = np.random.normal(0, var_v)  # gaussian observation noise with mean 0 variance var_v
        Z = C @ X + V  # observation
        L_obs.append(Z)  # stores the new observation in the observation list
        X_hat = A @ X_hat + K * (Z - Z_hat)  # state estimate
        L_est.append(X_hat)  # stores the new estimation in the estimation list

    return L_state, L_obs, L_est

def compute_gradient(L_state, L_obs, L_est):  # same as compute Q_0
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



# K = np.array([[2.0], [2.0]])
# L_state, L_obs, L_est = generate_path(K, 1)
# print(compute_gradient(L_state, L_obs, L_est))

