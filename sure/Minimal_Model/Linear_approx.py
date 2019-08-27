# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Some tools to determine function value based on linear approximation has been included here
import numpy as np
"""not useful any more"""

def Linear_interpolation(t, I_list, t_list):
    """ Find out I at time t using a linear interpolation between two points in t_list
    @:param t:         given time
    @:param I_list:    a list of function values with respect to t
    @:param t_list:    the corresponding time discretization """

    # np.searchsorted - return i satisfies t_list[i-1] <= t < t_list[i]
    idx1 = np.searchsorted(t_list, t, side="right") - 1
    idx2 = idx1+1

    if idx2<len(I_list):
        y1   = I_list[idx1]
        y2   = I_list[idx2]
        x1   = t_list[idx1]
        x2   = t_list[idx2]
        I_t  = (y2-y1)/(x2-x1)*(t-x1)+y1
        return I_t
    return I_list[-1]


def Constant_interpolation(t, I_list, t_list):
    """ Find out function I at time t using a piecewise constant approximation between two points in t_list
        @:param t:         given time
        @:param I_list:    a list of function values with respect to t
        @:param t_list:    the corresponding time discretization """

    idx1 = np.searchsorted(t_list, t, side="right") - 1 # t_list[idx1]<= t < t_list[idx1+1]
    return I_list[idx1]
