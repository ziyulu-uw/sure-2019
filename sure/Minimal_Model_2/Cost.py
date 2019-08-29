# Author: Xinyu Li, Ziyu Lu
# Email: xl1796@nyu.edu, zl1546@nyu.edu
# Date: August 2019
# Description: The cost is calculated here

import numpy as np

def cost_computation(G, X, I, Ra, Gb, Ib, control_gain, lbda=1):
    """
    a function to calculate the total cost from time 0 to t1
    :param G, X, I, Ra: list of state variables of the whole run
    :param Gb, Ib:      basal values
    :param control_gain = [h1,h2,h3,h4]
    :return: cost of the whole run
    """

    G_hat = np.zeros(len(G))
    for i in range(len(G)):
        if G[i] > 140:  # higher than the upper bound
            G_hat[i] = G[i] - 140
        elif G[i] < 80:  # lower than the lower bound
            G_hat[i] = 80 - G[i]
        else:  # within the euglycemic zone
            G_hat[i] = 0

    h1, h2, h3, h4 = control_gain
    vn_list = h1*(G-Gb) + h2*X + h3*(I-Ib) + h4*Ra
    J = 1/(2*len(G))*(np.sum(G_hat**2) + lbda*np.sum(vn_list**2))
    """lbda = 50 decided in the paper"""

    return J
