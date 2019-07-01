# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program generates paths for the Kalman filter optimization problem

import Initialization as init
import numpy as np


def generate_path(K, z):
    # K -- Kalman gain, z -- random seed (set when testing)
    # generates a path from the initial state X0 with Kalman gain K
    # returns a list of states, a list of observations, and a list of state estimations from the path

    # np.random.seed(z)  # set random seed for testing purpose
    X = init.X0  # initial state
    X_hat = init.X0  # initial state estimate
    L_state = []  # a list that stores the states in a path
    L_obs = []  # a list that stores the observations in a path
    L_est = []  # a list that stores the state estimations in a path
    L_state.append(X)
    L_obs.append(0)
    L_est.append(X_hat)

    for n in range(init.N):

        Z_hat = init.C @ init.A @ X_hat  # predicted observation
        W = np.random.multivariate_normal([0,0], init.R)  # gaussian system noise with mean 0 covariance R
        W = np.array(W, ndmin=2)
        W = np.transpose(W)
        X = init.A @ X + W  # state update
        L_state.append(X)  # stores the new state in the state list
        V = np.random.normal(0, init.Q)  # gaussian observation noise with mean 0 variance Q
        Z = init.C @ X + V  # observation
        L_obs.append(Z)  # stores the new observation in the observation list
        X_hat = init.A @ X_hat + K * (Z - Z_hat)  # state estimate
        L_est.append(X_hat)  # stores the new estimation in the estimation list

    # print("States", L_state)
    # print("Obs", L_obs)
    # print("Est", L_est)

    return L_state, L_obs, L_est

