# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This module is used for plot some functions

import matplotlib.pyplot as plt
from Glucose_ODE_Solver import glucose_solver
from Glucose import  find_Ra_list
from Insulin import find_I_list
from Insulin_ODE_Solver import insulin_solver
#import numpy as np

def Plot_Ra(t_list, Ra_list):
    """a function used to plot glucose rate Ra
    @:param t_list:   time discretization
    @:param Ra_list:  glucose disappearance rate """

    plt.plot(t_list, Ra_list)
    plt.title("Glucose Rate of Appearance $R_a$")
    plt.xlabel("min")
    plt.ylabel("$R_a$ (mg/kg/min)")
    plt.show()
    return Ra_list


def Plot_I(t_list, I_list):
    """a function used to plot glucose rate Ra
    @:param t_list:   time discretization
    @:param I_list:   insulin function """

    plt.plot(t_list,I_list)
    plt.title("Insulin $I(t)$")
    plt.xlabel("min")
    plt.ylabel("$I$ (mU/l)")
    plt.show()


def Plot_G(t_list, G):
    """a function used to plot glucose rate Ra
    @:param t_list:   time discretization
    @:param I_list:   insulin function """

    plt.plot(t_list,G,'-')
    plt.title("Glucose $G(t)$")
    plt.xlabel("min")
    plt.ylabel("$G$ (mg/dl)")
    plt.show()


def Plot_X(t_list, sol):
    plt.plot(t_list,sol[0],'-')
    plt.title("$X(t)$")
    plt.xlabel("min")
    plt.ylabel("$X$ (mg/dl)")
    plt.show()


def test_I(vn_list, I0, t_list, c1, c2, meas_time, dt):
    """ use python ODE package to test the theoretical solution is correct
    @:param vn_list:    a given control exerted on the insulin
    @:param I0:         initial condition of insulin level
    @:param t_list:     time discretization for the simulation
    @:param vn_list:    a list of insulin control
    @:param c1:         constant parameter in the insulin ODE model
    @:param c2:         constant parameter in the insulin ODE model """

    ## Theoretical solution
    I_list = find_I_list(t_list, vn_list, I0, c1, c2, meas_time, dt)
    ## ODE solver solution
    #print("*********************************************************************")
    #print("t_list:",len(t_list),"vn_list",len(vn_list))
    I     = insulin_solver(I0, t_list, vn_list, c1, c2)

    plt.plot(t_list, I_list,  label="theoretical solution")
    plt.plot(t_list, I, '--', label="numerical solution")
    plt.title("Insulin Level")
    plt.xlabel("min")
    plt.ylabel("$I$ (mU/l)")
    plt.legend()
    plt.show()

def test_Ra(t_list, tau, tk_list, qk_list, Ra_0):
    """ use python ODE package to test the theoretical solution is correct
    @:param t_list:     time discretization for the simulation
    @:param tau:        a constant parameter associated with digestion efficiency
    @:param tk_list:    a list of time t when the meal intake happens
    @:param qk_list:    a list of input value q representing the input glucose level
    @:param Ra_0:       initial condition of Ra(t)"""

    ## Theoretical solution
    Ra      = find_Ra_list(t_list, tk_list, qk_list, tau, Ra_0)
    ## ODE solver solution
    #t_list2 = np.linspace(0, 1000, 1000001)
    Ra_     = glucose_solver(t_list, tau, tk_list, qk_list, Ra_0)


    plt.plot(t_list,  Ra, label="theoretical solution")
    plt.plot(t_list, Ra_, '--', label="numerical solution")
    plt.title("Glucose Rate of Appearance $R_a$")
    plt.xlabel("min")
    plt.ylabel("$R_a$ (mg/kg/min)")
    plt.legend()
    plt.show()
