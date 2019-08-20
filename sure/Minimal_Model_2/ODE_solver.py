# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This program is a differential equation solver for minimal model

#import numpy as np
from scipy.integrate import odeint
from Glucose import Ra
from Linear_approx import Linear_func
#import Insulin_ODE_Solver
#import matplotlib.pyplot as plt

def dG_dX(x, t, p1, p2, p3, Gb, Ib, tk_list, qk_list, tau, t_list, I_list, Ra_0):
    """ ODE for minimal model
    @:param x:          [G, X]
    @:param t:          time
    @:param p1, p2, p3: bio parameters (unit: min^-1, min^-1, min^-1 mU/l)
    @:param Gb, Ib:     basal plasma glucose (mmol/l) and insulin (mU/l)
    @:param tk_list:    a list of meal time
    @:param qk_list:    a list of glucose input from meal
    @:param tau:        a constant parameter associated with digestion efficiency
    @:param t_list:     time discretization for the simulation
    @:param I_list:     insulin level
    """
    I = Linear_func(t, I_list, t_list)
    Ra_ = Ra(t, tk_list, qk_list, tau, Ra_0)
    G = x[0]
    X = x[1]
    dG_dt = -(p1 + X) * G + p1 * Gb + Ra_
    dX_dt = -p2 * X + p3 * (I - Ib)
    ## debug
    #if  t<56 and t>54:
        #print(t,G,Gb,G-Gb)
        #print("t:",t,"-p1(G-Gb):",-p1*(G-Gb), " -GX:",-G*X, " G:",G, " dG:", dG_dt)
        #print("Ra:", Ra(t), "p1*Gb", p1*Gb, )
        #print("t:",t, "I:",I)
        #print("I_list:",I_list)
        #print("t_list:",t_list)
        #print()

        #print("t:", t, "X:",X, "dX_dt", dX_dt)
    return [dG_dt, dX_dt]


def Minimod_ODE_solver(I_list, G0, X0, t_list, p1, p2, p3, Gb, Ib, tk_list, qk_list, tau, Ra_0):
    """ ODE for minimal model
    @:param I_list      insulin dynamics simulated by insulin solver or theoretical solution
    @:param G0:         initial condition of glucose level
    @:param t_list:     time discretization for the simulation
    @:param p1, p2, p3: bio parameters (unit: min^-1, min^-1, min^-1 mU/l)
    @:param Gb, Ib:     basal plasma glucose (mmol/l) and insulin (mU/l)
    @:param tk_list:    a list of meal time
    @:param qk_list:    a list of glucose input from meal
    @:param tau:        a constant parameter associated with digestion efficiency
    @:param vn_list:    a list of insulin control
    @:param I0:         initial condition of insulin level
    @:param c1:         constant parameter in the insulin ODE model
    @:param c2:         constant parameter in the insulin ODE model
    @:param meas_time:  time when measurements happen
    @:param dt:         time interval between 2 measurements"""

    ## initial condition for [G, X]
    x0 = [G0, X0]

    ## Find the values of Insulin
    #print("I0:",I0)

    #print(t_list)
    #print(I_list)
    #print(len(t_list))
    #print(len(vn_list))
    #print()
    ## Solve ODE system
    x      = odeint(dG_dX, x0, t_list, args=(p1, p2, p3, Gb, Ib, tk_list, qk_list, tau, t_list, I_list, Ra_0))

    ## return the result
    G = x[:,0]
    X = x[:,1]

    return G, X

