# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: The cost is calculated here

import numpy as np
from Insulin_Input import phar_kin
from Parameters import time_param

def cost_computation(I_t, Ib, G, Gb, r):
    """a function to calculate the total cost from time 0 to t1
    :param I_t the value of insulin I(t) by simulation
    :param Ib  basal insulin value
    :param G   simulation result of glucose function
    :param Gb  basal glucose value
    :param r   coefficient that determine the ratio of two parts in cost function
    :return total cost"""

    dt_I = time_param.t1/(len(I_t)-1)
    dt_G = time_param.t1/(len(G) - 1)
    return  r*np.sum((I_t-Ib)**2 * dt_I)  + np.sum((G-Gb)**2 * dt_G)
