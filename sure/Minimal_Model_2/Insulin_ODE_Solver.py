# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This program contains parameters and functions to determine I(t) using numpy ODE solver

from scipy.integrate import odeint
from Linear_approx import Linear_func


def dIdt(x, t, c1, c2, vn_list, t_list):
    """ ODE for insulin dynamics model
    @:param x:          [I]
    @:param t:          time
    @:param c1, c2:     constant parameter in the insulin ODE model
    @:param t_list:     time discretization for the simulation
    @:param vn_list:    a list of insulin control """

    I_ = x[0]
    dI_dt = -c1 * I_ + c2 * Linear_func(t, vn_list, t_list)    # function 'I' could also be used on vn_list

    return [dI_dt]


def insulin_solver(I0, t_list, vn_list, c1, c2):
    """ ODE solver for insulin dynamics
    @:param I0:         initial condition of insulin level
    @:param t_list:     time discretization for the simulation
    @:param vn_list:    a list of insulin control
    @:param c1:         constant parameter in the insulin ODE model
    @:param c2:         constant parameter in the insulin ODE model """

    ## initial condition for [G, X]
    x0 = [I0]

    ## Solve ODE system
    x  = odeint(dIdt, x0, t_list, args=(c1, c2, vn_list, t_list))

    ## return the result
    I_ = x[:,0]

    return I_
