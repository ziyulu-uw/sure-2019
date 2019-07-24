# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program computes the loss F

import numpy as np


def compute_loss(X, U, N, r):
    # X -- list of states from one path (X0, X1, ..., XN), 
    # U -- list of controls from one path (U0,U1,...,UN=0), 
    # Caution: U_N should be set as zero manually before call the function
    # N -- number of total time steps, r -- scaling factor
    # computes the mean-squared error: F = 1/2N \sum_{n=0}^{N} (X_n^TX_n + rU_n^TU_n), U_N=0
    # returns F

    F = 0
    assert (N == len(X) - 1), "Number of intended time steps and number of states not equal. Something is wrong."
    for i in range(N + 1):
        x = X[i, :, :]
        u = U[i, :, :]
        err1 = x.transpose() @ x
        err2 = u.transpose() @ u
        F += err1[0][0] + r*err2[0][0]

    return F / (2 * N)
