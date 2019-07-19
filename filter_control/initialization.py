# Author: Ziyu Lu, Xinyu Li
# Email: zl1546@nyu.edu, xl1796@nyu.edu
# Date: July 2019
# Description: This program defines all constant coefficients in the optimal filtering problem

import math
import cmath
import numpy as np
from numpy import linalg as LA


# constants
m = 1   # mass of the object
k = 0.5  # spring constant
gamma = 0.1  # friction coefficient
omega = k/m
mu = gamma/m
w = math.sqrt(omega - 0.25*mu**2)
N = 4  # number of time steps in one simulation
dt = 0.5  # step size in one simulation
sigma = 1  # noise coefficient in SDE
X0 = np.array([[1.0], [0.0]])
# X0 = np.array([1.0, 0.0])  # transpose of initial state
C = np.array([[1.0, 0.0]])  # observation matrix
B = np.array([[0.0], [1.0]])  # control coefficient matrix
S = [[0.5]]  # observation noise covariance
r = 1.0  # scaling factor in the cost
d_X = 2  # dimension of state
d_Z = 1  # dimension of observation
d_U = 1  # dimension of control


lambda1 = complex(-0.5*mu,w)
lambda2 = complex(-0.5*mu,-w)

# construct state transition matrix A
A11 = lambda2*cmath.exp(lambda1*dt) - lambda1*cmath.exp(lambda2*dt)
A12 = -cmath.exp(lambda1*dt) + cmath.exp(lambda2*dt)
A21 = lambda1*lambda2*(cmath.exp(lambda1*dt) - cmath.exp(lambda2*dt))
A22 = -lambda1*cmath.exp(lambda1*dt) + lambda2*cmath.exp(lambda2*dt)
A = np.array([[A11, A12], [A21, A22]])
A = np.divide(A, lambda2-lambda1)  # A turns out to be real
A = A.real

# print(A)

'''
# Checks the stability of the state dynamics
eigval, eigvec = LA.eig(A)
# print(eigval)
E = np.absolute(eigval)
# print(E)
for i in E:
    if i > 1:
        print("State dynamics is unstable")
        break
print("State dynamics is stable")
'''

# construct system noise covariance matrix R
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


