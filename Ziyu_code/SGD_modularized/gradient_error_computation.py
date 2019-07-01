# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program implements the backward gradient computation and computes the error for a given path

import Initialization as init
import numpy as np


def compute_gradient(K, L_state, L_obs, L_est):
    # L_state -- list of states from one path, L_obs -- list of observations from one path, \
    # L_est -- list of state estimations from one path
    # computes dF/dK, where F = 1/2N *\sum_{n=1}^N (X_hat_n-X_n)^2
    # returns dF/dK

    Q = 0  # start with Q_N = [0, 0]
    P = 2*np.transpose(L_est[init.N] - L_state[init.N])  # start with P_N = 2*(\hat{X_N} - X_N)^T
    for i in range(init.N-1, -1, -1):  # move backward
        diag = L_obs[i+1] - init.C @ init.A @ L_est[i]
        diag = np.array([1, 1]) * diag[0][0]
        Q = Q + P @ np.diag(diag)
        P = 2*np.transpose(L_est[i] - L_state[i]) + P @ (np.identity(2) - K @ init.C) @ init.A

    return Q/(2*init.N)


def compute_error(L_state, L_est):
    # L_state -- list of states from one path, L_est -- list of state estimations from one path
    # computes the mean-squared error: F = 1/2N *\sum_{n=1}^N (X_hat_n-X_n)^2
    # returns F

    F = 0
    assert (init.N == len(L_state)-1), "Number of intended time steps and number of states not equal. Something is wrong."
    for i in range(1, len(L_state)):
        error = L_state[i] - L_est[i]
        F += (error[0][0]**2 + error[1][0]**2)
    return F/(2*init.N)