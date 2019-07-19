# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program computes the loss F

import numpy as np


def compute_loss(X, U, N, r):
    # X -- list of states from one path, U -- list of controls from one path, \
    # N -- number of total time steps, r -- scaling factor
    # computes the mean-squared error: F = 1/2N *(X_N^TX_N + \sum_{n=0}^{N-1} (X_n^TX_n + rU_n^TU_n))
    # returns F

    F = 0
    assert (N == len(X) - 1), "Number of intended time steps and number of states not equal. Something is wrong."
    for i in range(N + 1):
        x = X[i, :, :]
        u = U[i, :, :]
        err1 = x.transpose() @ x
        err2 = u.transpose() @ u
        F += err1[0][0] + err2[0][0]

    return F / (2 * N)
