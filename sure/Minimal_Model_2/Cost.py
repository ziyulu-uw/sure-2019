# Author: Xinyu Li, Ziyu Lu
# Email: xl1796@nyu.edu, zl1546@nyu.edu
# Date: August 2019
# Description: The cost is calculated here

import numpy as np


def cost_computation(true_state, model_state_variable, Gb, Ib, control_gain, lbda, r, ub=140, lb=80):
    """
    a function to calculate the total cost from time 0 to t1
    :param true_G:      a true G list of the human, not the ODE model
    :param G, X, I, Ra: list of state variables of the whole run
    :param Gb, Ib:      basal values
    :param control_gain = [h1,h2,h3,h4]
    :param lbda: weight of the control in the cost
    :param r: weight of control cost
    :param ub, lb: upper and lower bound of the euglycemic zone
    :return: cost of the whole run
    """
    G,X,I,Ra = model_state_variable
    true_G, true_X, true_I, true_Ra = true_state
    # true_G = true_state[0]
    G_hat = np.zeros(len(G))
    for i in range(len(G)):
        if true_G[i] > ub:  # higher than the upper bound
            G_hat[i] = true_G[i] - ub
        elif G[i] < lb:  # lower than the lower bound
            G_hat[i] = 10*(lb - true_G[i])
        else:  # within the euglycemic zone
            G_hat[i] = 0

    h1, h2, h3, h4 = control_gain
    vn_list = h1*(G-Gb) + h2*X + h3*(I-Ib) + h4*Ra

    # J1 = np.sum(((G-np.mean(G))/np.std(G)-(true_G-np.mean(true_G))/np.std(true_G))**2)\
    #      + np.sum(((X-np.mean(X))/np.std(X)-(true_X-np.mean(true_X))/np.std(true_X))**2)\
    #      + np.sum(((I-np.mean(I))/np.std(I)-(true_I-np.mean(true_I))/np.std(true_I))**2)\
    #      + np.sum(((Ra-np.mean(Ra))/np.std(Ra)-(true_Ra-np.mean(true_Ra))/np.std(true_Ra))**2)  # data estimation cost
    # vn = np.array(vn_list)
    # J2 = np.sum(((G_hat - np.mean(G_hat)) / np.std(G_hat))**2) + \
    #      lbda*np.sum(((vn - np.mean(vn)) / np.std(vn))**2)  # control cost
    # J = J1+r*J2

    J_ = 1000 * np.sum(G_hat ** 2) + lbda * np.sum(vn_list ** 2) + np.sum((G - true_G) ** 2) + np.sum((Ra - true_state[3]) ** 2)\
         + np.sum((10 * (X - true_state[1])) ** 2) + np.sum((I - true_state[2]) ** 2)
    J = 1/(2*len(G))*J_

    return J

