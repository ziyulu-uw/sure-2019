# %load ODE_solver
# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This program is a differential equation solver for minimal model

from scipy.integrate import odeint
from Linear_approx import Constant_interpolation

def dMiniMod(x, t, p1, p2, p3, Gb, Ib, tau, c1, c2, vn_list, t_list, tk_list, qk_list, meal_time):
    """ ODE for minimal model
    @:param p1, p2, p3: bio parameters (unit: min^-1, min^-1, min^-1 mU/l)
    @:param Gb, Ib:     basal plasma glucose (mmol/l) and insulin (mU/l)
    @:param tau:        a constant parameter associated with digestion efficiency
    @:param c1:         constant parameter in the insulin ODE model
    @:param c2:         constant parameter in the insulin ODE model
    @:param vn_list:    a list of control exerted on the model
    @:param t_list:     time descritization for the whole simulation
    @:param tk_list:    a list of meal time
    @:param qk_list:    a list of glucose input from meal
    @:param meal_time:  how long a meal intake last"""

    G  = x[0]
    X  = x[1]
    I  = x[2]
    Ra = x[3]
    vn = Constant_interpolation(t, vn_list, t_list)  # function linear_func could also be used on vn_list

    dG_dt = -(p1 + X) * G + p1 * Gb + Ra
    dX_dt = -p2 * X + p3 * (I - Ib)
    dI_dt = -c1 * I + c2 * vn
    dRa_dt= -1/tau*(Ra - D(t, tk_list, qk_list, meal_time))

    return [dG_dt, dX_dt, dI_dt, dRa_dt]


def Minimod_ODE_solver(G0, X0, I0, Ra_0, sub_t_list, p1, p2, p3, Gb, Ib, tau, c1, c2, vn_list,t_list, tk_list, qk_list, meal_time):
    """ ODE for minimal model
    @:param G0:         initial condition of glucose level
    @:param X0:         initial condition of remote insulin level
    @:param I0:         initial condition of insulin level
    @:param Ra_0:       initial condition of glucose appearance rate
    @:param sub_t_list: time discretization for the simulation
    @:param p1, p2, p3: bio parameters (unit: min^-1, min^-1, min^-1 mU/l)
    @:param Gb, Ib:     basal plasma glucose (mmol/l) and insulin (mU/l)
    @:param tau:        a constant parameter associated with digestion efficiency
    @:param c1:         constant parameter in the insulin ODE model
    @:param c2:         constant parameter in the insulin ODE model
    @:param vn_list:    a list of control exerted on the model
    @:param t_list:     time descritization for the whole simulation
    @:param tk_list:    a list of meal time
    @:param qk_list:    a list of glucose input from meal
    @:param meal_time:  how long a meal intake last"""

    ## initial condition for [G, X]
    x0     = [G0, X0, I0, Ra_0]

    ## Solve ODE system
    x      = odeint(dMiniMod, x0, sub_t_list, args=(p1, p2, p3, Gb, Ib, tau, c1, c2, vn_list, t_list, tk_list, qk_list, meal_time))

    ## return the result
    G  = x[:,0]
    X  = x[:,1]
    I  = x[:,2]
    Ra = x[:,3]

    return G, X, I, Ra

