# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This program contains parameters and functions to determine Glucose function theoretically

import numpy as np

def u(t, tk):
    """a step function u: u(t) = 0 if t<=tk, u(t) = 1 if t> tk
    @:param t:  time
    @:param tk: meal time
    @:return u(t) """

    return (t>tk)*1

'''Ra is normally from 5, around 7.5, up to 12.5, 15
could be 30-40 after meal'''
def inner_Ra(t, tk, qk, tau):
    """Ra is a sum of inner_Ra
    @:param t:  time
    @:param tk: meal time
    @:param qk: glucose input from meal
    @:param tau:  a constant parameter associated with digestion efficiency"""
    u_tk = u(t, tk)
    if u_tk == 0:
        return 0
    return u_tk*qk/tau*np.exp(-1/tau*(t-tk))


def Ra(t, tk_list, qk_list, tau, Ra_0):
    """I(t) is the insulin function at (tn, tn + delta t), a theoretical solution of insulin ODE give I(tn)
    @:param tk_list:  a list of time t when the meal intake happens
    @:param qk_list:  a list of input value q representing the input glucose level    ( len(qk_list)=len(tk_list) )
    @:param tau:      a constant parameter associated with digestion efficiency
    @:return the value of Ra at time t"""

    Ra_t = Ra_0
    for i in range(len(tk_list)):
        Ra_t += inner_Ra(t, tk_list[i], qk_list[i], tau)

    return Ra_t

def find_Ra_list(t_list, tk_list, qk_list, tau, Ra_0):
    """a function used to plot glucose rate Ra
    @:param t_list:   time discretization
    @:param tk_list:  meal intake time list
    @:param qk_list:  meal intake value list
    @:param tau:      digestion coefficient"""

    Ra_list = []
    for t in t_list:
        Ra_list.append(Ra(t, tk_list, qk_list, tau, Ra_0))
    return Ra_list
