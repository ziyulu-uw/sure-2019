# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This program simulate the process with filter
from Insulin_ODE_Solver import insulin_solver
from ODE_solver import Minimod_ODE_solver
import numpy as np

def path_generator(vn_list, p1, p2, p3, Gb, Ib, I0, c1, c2, meas_time, t_list, tk_list, qk_list, tau, G0,X0, Ra_0, dt, h):
    """this function generates the path given control vn_list
    @:param vn_list:                         a list of controls. dimension = len(t_list)
    @:param p1, p2, p3, Gb, Ib, I0, c1, c2:  bio parameters
    @:param meas_time:                       the time where measurement happens
    @:param t_list:                          time discretization
    @:param tk_list, qk_list:                food intake time and food intake value
    @:param tau:                             digestion parameter
    @:param G0,X0,Ra_0:                      initial conditions of glucose level, remote insulin level, glucose disappearance rate
    @:param dt, h:                           time step between two measurements, time step of time discretization"""

    Gn         = G0
    Xn         = X0
    G_list     = [G0]
    X_list     = [X0]
    N_meas     = len(meas_time)
    I_list     = insulin_solver(I0, t_list, vn_list, c1, c2)

    for i in range(N_meas-1):
        ## Discretize to sub section in one dt
        sub_t_list  =  t_list[int(i*dt/h): int((i+1)*dt/h+1)]
        sub_I_list  =  I_list[int(i*dt/h): int((i+1)*dt/h+1)]

        ## Solve the ODE in the sub section
        G, X       =   Minimod_ODE_solver(sub_I_list, Gn, Xn, sub_t_list, p1, p2, p3, Gb, Ib, tk_list, qk_list, tau, Ra_0)

        ## Update Gn as the initial condition for the next iteration
        Gn  = G[-1]
        Xn  = X[-1]

        ## Record
        G_list.extend(list(G[1:]))
        X_list.extend(list(X[1:]))

    return np.array(G_list), np.array(X_list), np.array(I_list)

def cont_path_generator(vn_list, p1, p2, p3, Gb, Ib, I0, c1, c2, t_list, tk_list, qk_list, tau, G0, X0, Ra_0):
    """this function will NOT be used in later model
    It is only used to test that out current discrete ODE calculation is correct
    Later, only the function above will be used"""

    I_list = insulin_solver(I0, t_list, vn_list, c1, c2)

    G, X   = Minimod_ODE_solver(I_list, G0, X0, t_list, p1, p2, p3, Gb, Ib, tk_list, qk_list, tau, Ra_0)
    return np.array(G), np.array(X), np.array(I_list)
