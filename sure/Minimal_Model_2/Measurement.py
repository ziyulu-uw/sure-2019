# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: An in-silico advance person is generated here, by adding the observation noise, we can get the measurements

import numpy as np
import matplotlib.pyplot as plt
from Path_unit import generate_path_unit

def advance_person(true_init_cond, control_gain, N_meas, sim_idx, T, T_list, noise):
    """simulate the real person, with that person’s physical parameters
    take input from the controller and use the noise path to advance by time T and generate a new
    (noisy) observation
    @:param true_init_cond    true initial condition for the human subject
    @:param control_gain = [h1,h2,h3,h4]:    control of the whole run: h1(G-Gb)+h2(X)+h3(I-Ib)+h4(Ra)
    @:param N_meas:           the number of measurements
    @:param sim_idx:          represent which control simulation it is
    @:param T_list:           time discretization in one control simulation
    @:param T:                Time for one control period
    @:param noise:            noise of this control unit """

    ## Minimal Model Parameter to generate measurements
    p1 = 0.6  # param determined by bio-experiments (unit: min^-1)
    p2 = 0.0122  # unit: min^-1
    p3 = 1.7e-5  # unit: min^-2 mU/l
    Gb = 125  # basal plasma glucose (unit:mg/dl)

    ## Insulin Parameter
    c1 = 0.25  # unit: min^-1
    c2 = 0.2   # unit: min^-1
    Ib = 0

    # digestion coefficient
    tau      = 100             #unit: 1

    ## True meal model
    tk_list = np.array([60, 350, 720])   #unit: min
    # the time that the subject take the meal
    meal_time = 25
    # meal intake value
    qk_list = np.array([3000, 4500, 3500]) / meal_time  # unit: mg/min
    meal_params = (tk_list, qk_list, meal_time)

    ## Wrap the parameters into a list:
    param_list = [p1, p2, p3, tau, c1, c2]

    ## No Filter for the true subject!
    Filter = [0,0,0,0]          # we do not need filter when making up the person
    Z = np.zeros(len(T_list))   # Z in the generate_path_unit is used for filtering, so we do not need it neither

    G, X, I, Ra = generate_path_unit(true_init_cond, param_list, control_gain, Filter, Z, noise, Gb, Ib, sim_idx, N_meas, T, T_list, meal_params, idx=1)
    true_state = [G,X,I,Ra]
    true_meas = G
    obs_meas  = true_meas[1:] + noise[1][0]
    true_init_cond = [G[-1],X[-1],I[-1],Ra[-1]]
    return true_init_cond, obs_meas, true_state

def Plot_measurement(noisy_meas, t_list):
    plt.plot(t_list[:-1], noisy_meas,'o')
    plt.title("Glucose Measurements")
    plt.xlabel("min")
    plt.ylabel("$G$ (mg/dl)")
    plt.show()
