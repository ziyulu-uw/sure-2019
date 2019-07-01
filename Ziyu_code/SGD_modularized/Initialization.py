# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program defines the constant coefficients for the Kalman filter optimization problem

import math
import cmath
import numpy as np

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
Q = 0.1  # observation noise variance
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
