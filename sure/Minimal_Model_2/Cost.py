# Author: Xinyu Li, Ziyu Lu
# Email: xl1796@nyu.edu, zl1546@nyu.edu
# Date: August 2019
# Description: The cost is calculated here

import numpy as np


def cost_computation(true_G, model_state_variable, Gb, Ib, control_gain, ub=140, lb=80, lbda=0):
    """
    a function to calculate the total cost from time 0 to t1
    :param true_G:      a true G list of the human, not the ODE model
    :param G, X, I, Ra: list of state variables of the whole run
    :param Gb, Ib:      basal values
    :param control_gain = [h1,h2,h3,h4]
    :param ub, lb: upper and lower bound of the euglycemic zone
    :return: cost of the whole run
    """
    G,X,I,Ra = model_state_variable
    G_hat = np.zeros(len(G))
    for i in range(len(G)):
        if true_G[i] > ub:  # higher than the upper bound
            G_hat[i] = true_G[i] - ub
        elif G[i] < lb:  # lower than the lower bound
            G_hat[i] = lb - true_G[i]
        else:  # within the euglycemic zone
            G_hat[i] = 0

    h1, h2, h3, h4 = control_gain
    vn_list = h1*(G-Gb) + h2*X + h3*(I-Ib) + h4*Ra
    J = 1/(2*len(G))*(np.sum(G_hat**2) + lbda*np.sum(vn_list**2))

    return J
