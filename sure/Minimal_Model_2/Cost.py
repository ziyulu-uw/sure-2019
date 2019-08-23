# Author: Xinyu Li, Ziyu Lu
# Email: xl1796@nyu.edu, zl1546@nyu.edu
# Date: August 2019
# Description: The cost is calculated here

import numpy as np

def cost_computation(G, vn_list):
    """a function to calculate the total cost from time 0 to t1
    :param G     simulation result of glucose function
    :param vn_list:    a list of control exerted on the model
    :return cost in the current prediction and control horizon"""

    G_hat = np.zeros(len(G))
    for i in range(len(G)):
        if G[i] > 140:  # higher than the upper bound
            G_hat[i] = G[i] - 140
        elif G[i] < 80:  # lower than the lower bound
            G_hat[i] = 80 - G[i]
        else:  # within the euglycemic zone
            G_hat[i] = 0

    J = 1/len(G)*(np.sum(G_hat**2) + np.sum(vn_list**2)*50)
    """lambda is 50 decided in the paper"""

    return J
