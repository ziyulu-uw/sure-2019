# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: The cost is calculated here

import numpy as np

def cost_computation(G, Gb, lam, vn_list, end_time):
    """a function to calculate the total cost from time 0 to t1
    :param I_t   the value of insulin I(t) by simulation
    :param Ib    basal insulin value
    :param G     simulation result of glucose function
    :param Gb    basal glucose value
    :param lam   coefficient that determine the ratio of two parts in cost function
    :return total cost"""


    dt_G = end_time/(len(G) - 1)
    return lam*np.sum(vn_list)  + np.sum((G-Gb)**2 * dt_G)
