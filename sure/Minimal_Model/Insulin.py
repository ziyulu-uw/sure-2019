# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This program contains parameters and functions to determine I(t) theoretically
"""this page has been given up"""
import numpy as np

def find_tn(tn_list, dt, t):
    """given a list of time descritization and time t, find out tn such that
    tn < t <= t_{n+1}"""
    tn = tn_list[int(t//dt)]
    if t%dt != 0 or t==0:
        return tn
    return tn-dt


def inner_I(t, In, vn, c1, c2, meas_time, dt):
    """inner_I(t) is the insulin function at (tn, tn + delta t),
       a theoretical solution of insulin ODE give I(tn)
    @:param t:         time
    @:param In:        initial condition for I(t) at (tn, tn + delta t)
    @:param vn:        a constant control exerted on insulin
    @:param c1:        constant parameter in the insulin ODE model
    @:param c2:        constant parameter in the insulin ODE model
    @:param meas_time: time when measurements happen
    @:param dt:        time interval between 2 measurements
    @:return           the value of I at time t"""

    ## Find the previous state time tn
    tn  = find_tn(meas_time, dt, t)
    ## Find I(t) for t in (tn, tn+dt)
    I_t = c2*vn/c1*(1-np.exp(-c1*(t-tn)))+In*np.exp(-c1*(t-tn))
    return I_t


def vn_conversion(vn_list, dt, h):
    """because the number of control should be the number of measurement-1, but
    sometimes we need number of control to be same as the number of time steps.
    This function convert vn_list from len(meas_time) to len(t_list)"""

    step = int(dt//h)
    v = np.reshape(vn_list, [len(vn_list), 1, 1])
    b = np.ones([len(vn_list),step , 1])
    c = b@v
    new_vn_list = np.reshape(c, [1, step*len(vn_list)])
    return new_vn_list[0]


def find_I_list(t_list, vn_list, I0, c1, c2, meas_time, dt):
    """a function find out I(t) at any given time t
    @:param t_list:    time discretization
    @:param vn_list:   a list of control
    @:param I0:        initial value of Insulin
    @:param c1:        insulin coefficient
    @:param c2:        insulin coefficient
    @:param meas_time: time when measurements happens
    @:param dt:        time interval between two measurements happens"""

    I_list  = []
    In      = I0
    for i in range(len(t_list)):
        t_  = t_list[i]
        vn  = vn_list[i]
        I_t = inner_I(t_, In, vn, c1, c2, meas_time, dt)
        I_list.append(I_t)
        if t_ % dt == 0:
            In = I_t
    return I_list

def Linear_func(t, I_list, t_list):
    """ Find out I at time t using a linear approximation between two points in t_list
    @:param t:         given time
    @:param I_list:    a list of insulin values with respect to t
    @:param t_list:    the corresponding time discretization """

    idx1 = int(t//(t_list[1] - t_list[0]))
    idx2 = idx1+1

    if idx2<len(I_list):
        y1   = I_list[idx1]
        y2   = I_list[idx2]
        x1   = t_list[idx1]
        x2   = t_list[idx2]
        I_t  = (y2-y1)/(x2-x1)*(t-x1)+y1
        return I_t
    return I_list[-1]
