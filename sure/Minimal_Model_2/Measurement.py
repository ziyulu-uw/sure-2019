# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Generate noisy measurements here

import numpy as np
import matplotlib.pyplot as plt
from Simulation import path_generator

def measurement_generation(vn_list, t_list,  dt,h, meas_time):
    """a function in order to generate noisy measurements
    @:param t_list:           time discretization
    @:param tk_list, qk_list: meal time and meal intake amount
    @:param dt:               interval between two measurements"""
    ## Minimal Model Parameter to generate measurements
    p1 = 0.6  # param determined by bio-experiments (unit: min^-1)
    p2 = 0.0122  # unit: min^-1
    p3 = 1.7e-5  # unit: min^-2 mU/l
    Gb = 125  # basal plasma glucose (unit:mg/dl)
    G0 = 130
    X0       = 0
    Ra_0     = 0
    ## Insulin Parameter
    c1 = 0.25  # unit: min^-1   [unknown parameter!!!]
    c2 = 0.2  # unit: min^-1   [unknown parameter!!!]
    I0 = 0.38  # unit: mU/l     [unknown initial condition]
    Ib = 0

    # digestion coefficient
    tau      = 100                  #unit: 1  [unknown parameter!!!]

    #variance of measurement noise
    var      = 10

    # true meal model
    tk_list = [60, 350, 720]  # unit: min
    # the time that the subject take the meal
    meal_time = 5
    # meal intake value
    qk_list = np.array([3000, 4500, 3500]) / meal_time  # unit: mg/min, from ref 1 in Xinyu's writeup

    G, X, I, Ra = path_generator(vn_list, p1, p2, p3, Gb, Ib, I0, c1, c2, meas_time, t_list, tau, G0,X0, Ra_0, dt, h, tk_list, qk_list, meal_time)

    model_meas = G[::int(dt/(t_list[1]-t_list[0]))]
    true_meas= model_meas+ np.random.normal(0, var, [len(model_meas)])
    return true_meas

def Plot_measurement(noisy_meas, t_list, dt):
    plt.plot(t_list[::int(dt/(t_list[1]-t_list[0]))], noisy_meas,'o')
    plt.title("Glucose Measurements")
    plt.xlabel("min")
    plt.ylabel("$G$ (mg/dl)")
    plt.show()
