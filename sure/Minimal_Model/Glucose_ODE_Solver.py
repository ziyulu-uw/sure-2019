# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This program contains parameters and functions to determine Ra(t) using numpy ODE solver
"""this page is not used in the program except for comparision to verify theoretical solution"""

from scipy.integrate import odeint
import numpy as  np

def D(t, tk_list, qk_list):
    """meal distrubance model: D(mg/kg/min)
    @:param t:          time
    @:param tk_list:    a list of time t when the meal intake happens
    @:param qk_list:    a list of input value q representing the input glucose level
                        ( len(qk_list)=len(tk_list) ) """
    D_t = 0
    for i in range(len(tk_list)):
        D_t += DiracDelta(t, tk_list[i])*qk_list[i]

    #debug...print(t, D_t)
    return D_t

def DiracDelta(t,q):
    """Dirac Delta function as the limit of normal distributions as a goes to 0
    @:param t: time
    @:param q: the peak point"""

    a = 1.3    #notice: though a should be as small as possible to represent a Dirac Delta function
               #however, if a is too small, it will cause difficulty to for ODE solver, it may miss some peaks
    return  1/(a*np.sqrt(np.pi))*np.exp(-((t-q)/a)**2)


def dRadt(x, t, tau, tk_list, qk_list):
    """ ODE for insulin dynamics model
    @:param x:          [I]
    @:param t:          time
    @:param tau:        a constant parameter associated with digestion efficiency
    @:param tk_list:    a list of time t when the meal intake happens
    @:param qk_list:    a list of input value q representing the input glucose level
                        ( len(qk_list)=len(tk_list) ) """

    Ra = x[0]
    dRa_dt = -1/tau * (Ra - D(t, tk_list, qk_list))

    return [dRa_dt]


def glucose_solver(t_list,  tau, tk_list, qk_list, Ra_0):
    """ ODE solver for glucose disappearance rate Ra(t)
    @:param t_list:     time discretization for the simulation
    @:param tau:        a constant parameter associated with digestion efficiency
    @:param tk_list:    a list of time t when the meal intake happens
    @:param qk_list:    a list of input value q representing the input glucose level
    @:param Ra_0:       initial condition of Ra(t)"""

    ## initial condition for [G, X]
    x0 = [Ra_0]

    ## Solve ODE system
    x  = odeint(dRadt, x0, t_list, args=(tau, tk_list, qk_list))

    ## return the result
    Ra = x[:,0]

    return Ra

