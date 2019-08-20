# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Generate noisy measurements here

import numpy as np
from ODE_solver import Minimod_ODE_solver
#from Initialization import meas_time, t_list, tk_list, qk_list, dt
import matplotlib.pyplot as plt
from Insulin_ODE_Solver import insulin_solver

def measurement_generation( t_list, tk_list, qk_list, dt):
    """a function in order to generate noisy measurements
    @:param t_list:           time discretization
    @:param tk_list, qk_list: meal time and meal intake amount
    @:param dt:               interval between two measurements"""
    ## Minimal Model Parameter to generate measurements
    p1       = 0.2 #param determined by bio-experiments (unit: min^-1)
    p2       = 0.122   #unit: min^-1
    p3       = 2e-4  #unit: min^-2 mU/l
    Gb       = 70      #basal plasma glucose (unit:mg/dl)
    G0       = 100
    X0       = 0
    Ra_0     = 0
    ## Insulin Parameter
    c1       = 0.01               #unit: min^-1   [unknown parameter!!!]
    c2       = 0.02               #unit: min^-1   [unknown parameter!!!]
    I0       = 0.38               #unit: mU/l     [unknown initial condition]
    Ib       = 0
    # digestion coefficient
    tau      = 80                  #unit: 1  [unknown parameter!!!]

    #variance of measurement noise
    var      = 10

    #control
    vn_list = np.zeros([1, len(t_list)])[0]
    vn_list[:100]=70
    vn_list[500:700]=80
    vn_list[1300:1500]=90

    I_list = insulin_solver(I0, t_list, vn_list, c1, c2)
    G, X   =  Minimod_ODE_solver(I_list, G0, X0, t_list, p1, p2, p3, Gb, Ib, tk_list, qk_list, tau, Ra_0)
    true_meas = G[::int(dt/(t_list[1]-t_list[0]))]
    noisy_meas= true_meas+ np.random.normal(0, var, [len(true_meas)])
    return noisy_meas

def Plot_measurement(noisy_meas, t_list, dt):
    plt.plot(t_list[::int(dt/(t_list[1]-t_list[0]))], noisy_meas,'o')
    plt.title("Glucose Measurements")
    plt.xlabel("min")
    plt.ylabel("$G$ (mg/dl)")
    plt.show()
