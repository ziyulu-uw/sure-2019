# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program provides initialization parameters.
#              But it allows to change dt or other params and give different initial A

import numpy as np

def init(k, gamma, dt, sigma, x0, v0, r, obv_noise):
    """:parameter k - spring constant
       :parameter gamma - friction coefficient
       :parameter dt - time discretization
       :parameter sigma - noise coefficient in SDE
       :parameter x0, v0 - initial condition
       :parameter r - scaling factor in the cost
       :parameter obv_noise - covariance of observation noise"""

    # constants
    m = 1   # mass of the object
    #k = 0.5  # spring constant
    #gamma = 0.1  # friction coefficient
    omega = k/m
    mu = gamma/m
    w = np.sqrt(omega - 0.25*mu**2)
    t1 = 60
    N = int(t1/dt+1)  # number of time steps in one simulation

    #dt = 0.1  # step size in one simulation
    t = np.linspace(0, t1, N)
    #sigma = 0.1  # noise coefficient in SDE
    #x0 = 1.0
    #v0 = 0.0
    X0 = np.array([[x0], [v0]])
    C = np.array([[1.0, 0.0]])  # observation matrix
    B = np.array([[0.0], [1.0]])  # control coefficient matrix
    S = np.array([[obv_noise]])  # observation noise covariance
    #r = 1.0  # scaling factor in the cost
    d_X = len(X0)  # dimension of state
    d_Z = 1  # dimension of observation
    d_U = 1  # dimension of control

    lambda1 = complex(-0.5*mu,w)
    lambda2 = complex(-0.5*mu,-w)

    # construct state transition matrix A
    A11 = lambda2*np.exp(lambda1*dt) - lambda1*np.exp(lambda2*dt)
    A12 = -np.exp(lambda1*dt) + np.exp(lambda2*dt)
    A21 = lambda1*lambda2*(np.exp(lambda1*dt) - np.exp(lambda2*dt))
    A22 = -lambda1*np.exp(lambda1*dt) + lambda2*np.exp(lambda2*dt)
    A = np.array([[A11, A12], [A21, A22]])
    A = np.divide(A, lambda2-lambda1)  # A turns out to be real
    A = A.real

    # construct system noise covariance matrix R
    e1 = np.exp(2*lambda1*dt)
    e2 = np.exp(2*lambda2*dt)
    e3 = np.exp((lambda1 + lambda2)*dt)
    R11 = 0.5*(e1-1)/lambda1 + 0.5*(e2-1)/lambda2 - 2*(e3-1)/(lambda1 + lambda2)
    R12 = 0.5*(e1-1) + 0.5*(e2-1) - (e3-1)
    R21 = R12
    R22 = lambda1*0.5*(e1-1) + lambda2*0.5*(e2-1) - 2*lambda1*lambda2*(e3-1)/(lambda1 + lambda2)
    R = np.array([[R11, R12], [R21, R22]])
    R = np.multiply(R, (sigma/(lambda2-lambda1))**2)  # R turns out to be real
    R = R.real

    return x0, v0, t, X0, A, B, C, S, R, d_X, d_U, d_Z, r, N
