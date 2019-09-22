# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This program is a differential equation solver for minimal model

from scipy.integrate import odeint
from Meal import D

def dMiniMod(x, t, param_list, vn, Gb, Ib, sim_idx, T, meal_params):
    """ ODE for minimal model
    @:param x           state variable
    @:param t           a time scalar

    param_list  = [p1, p2, p3, tau, c1, c2]
    @:param p1, p2, p3: bio parameters (unit: min^-1, min^-1, min^-1 mU/l)
    @:param tau:        a constant parameter associated with digestion efficiency
    @:param c1:         constant parameter in the insulin ODE model
    @:param c2:         constant parameter in the insulin ODE model

    @:param vn          constant control over 5 minutes
    @:param Gb, Ib:     basal plasma glucose (mmol/l) and insulin (mU/l)
    @:param T_list:     time descritization for one control simulation
    @:param tk_list:    a list of meal time
    @:param qk_list:    a list of glucose input from meal
    @:param meal_time:  how long a meal intake last"""

    ## get the state variable
    G, X, I, Ra = x

    ## get the parameters
    p1, p2, p3, tau, c1, c2  = param_list

    ## ODEs for the system
    dG_dt = -(p1 + X) * G + p1 * Gb + Ra
    dX_dt = -p2 * X + p3 * (I - Ib)
    dI_dt = -c1 * I + c2 * vn
    dRa_dt = -1 / tau * (Ra -  D(t, sim_idx, T, meal_params))

    ## Return the derivatives
    return [dG_dt, dX_dt, dI_dt, dRa_dt]


def Minimod_ODE_solver(init_cond, sub_t_list, param_list, vn, Gb, Ib, sim_idx, T, meal_params):
    """ ODE for minimal model: only last 5 min [time between two measurement]
    Given Gn, return Gn+1
    (return 4 floats)

    @:param: init_cond = [G0, X0, I0, Ra_0]
    @:param G0:         initial condition of glucose level, remote insulin level, insulin level
                        glucose appearance rate
    @:param sub_t_list: time discretization for the simulation
    @:param param_list: [p1, p2, p3, tau, c1, c2]
    @:param vn          constant control over 5 minutes
    @:param Gb, Ib:     basal plasma glucose (mmol/l) and insulin (mU/l)
    @:param T_list:     time descritization for one control simulation
    @:param tk_list:    a list of meal time
    @:param qk_list:    a list of glucose input from meal
    @:param meal_time:  how long a meal intake last"""

    ## initial condition for [G, X, I, Ra]
    x0 = init_cond

    ## Solve ODE system
    x = odeint(dMiniMod, x0, sub_t_list, args=(param_list, vn, Gb, Ib, sim_idx, T, meal_params))

    ## return the state at time n+1
    G = x[-1, 0]
    X = x[-1, 1]
    I = x[-1, 2]
    Ra= x[-1, 3]

    return [G, X, I, Ra]

