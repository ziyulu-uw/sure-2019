# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program checks whether the filter dynamics is stable given an initial Kalman gain

import numpy as np
from numpy import linalg as LA


def check_stability(A, C, K):
    # A -- state transition matrix, C -- observation matrix, K -- transpose of initial Kalman gain
    # checks if the filter dynamics is stable given the initial Kalman gain
    # returns True if stable, False if unstable

    K = np.array(K, ndmin=2).transpose()
    M = A - K @ C @ A
    eigval, eigvec = LA.eig(M)
    E = np.absolute(eigval)
    # print(E)
    for i in E:
        if i > 1:
            return False

    return True
