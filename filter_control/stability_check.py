# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program checks whether the filter dynamics is stable given an initial Kalman gain

import numpy as np
from numpy import linalg as LA


def check_stability(A, B, C, K, G, d_X):
    # A -- state transition matrix, B -- control coefficient matrix, C -- observation matrix, \
    # K -- initial Kalman gain, G -- initial control gain, d_X -- dimension of state
    # checks if the filter and control dynamics is stable given the initial Kalman gain and control gain
    # returns True if stable, False if unstable

    M = (np.identity(d_X) - K @ C) @ (A + B @ G)
    eigval, eigvec = LA.eig(M)
    E = np.absolute(eigval)
    # print(E)
    for i in E:
        if i > 1:
            return False

    return True
