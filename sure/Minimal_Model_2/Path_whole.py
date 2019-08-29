# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This program simulate the process of the whole run

from Measurement import advance_person
from Path_unit import generate_path_unit
import numpy as np

def generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params):
    """
    call advance_person and advance_controller N times (NT = total time) to get a path in state space

    @:param init_cond = G0,X0,I0,Ra_0        initial conditions of glucose level,
                                             remote insulin level, insulin level, glucose disappearance rate
    @:param param_list:                      [p1, p2, p3, tau, c1, c2]
    @:param control_gain = [h1,h2,h3,h4]:    control of the whole run: h1(G-Gb)+h2(X)+h3(I-Ib)+h4(Ra)
    @:param total_noise:                     noise of the whole run (process noise, observation noise)
    @:param Gb, Ib:                          basal values [known parameter]
    @:param N_meas:                          the number of measurements in one control simulation
    @:param T_list:                          time discretization in one control simulation
    @:param T:                               Time for one control period
    @:param meal_params                      meal time and intake amount
    @:param idx:                             If the function is called for ODE model, set idx to be zero, so that no process noise will be added;
                                             If the function is called for building the person, set idx to be one. Process noise will be added to correct the model.
    :return: G,X,I,Ra                        arrays of state variable for a whole run """

    G0, X0, I0, Ra_0 = init_cond

    G_list = [G0]
    X_list = [X0]
    I_list = [I0]
    Ra_list= [Ra_0]
    true_G_list = [G0]
    true_init_cond = init_cond.copy()
    meas_list = []

    for i in range(N):
        # take the part of noise corresponding to the control period out from the noise of the whole run
        process_noise = total_noise[0][:,i*N_meas:(i+1)*N_meas]  # [4, N_meas in one control unit]
        obv_noise = total_noise[1][:,i*N_meas:(i+1)*N_meas]      # [1, N_meas in one control unit]
        noise = (process_noise, obv_noise)

        ## Get the measurements in this control period
        true_init_cond, Z, true_G = advance_person(true_init_cond, control_gain, N_meas, i, T, T_list, noise)  # advance_person call generate_path_unit

        # run each control unit in a row, use the last state value of this period as the initial condition for next period
        G, X, I, Ra = generate_path_unit(init_cond, param_list, control_gain, Filter, Z, noise, Gb, Ib, i, N_meas, T, T_list, meal_params, idx=0)
        init_cond = [G[-1],X[-1],I[-1],Ra[-1]]

        ## Record
        G_list.extend(list(G[1:]))
        X_list.extend(list(X[1:]))
        I_list.extend(list(I[1:]))
        Ra_list.extend(list(Ra[1:]))
        true_G_list.extend(list(true_G[1:]))
        meas_list.extend(list(Z))

    model_state_variable = [np.array(G_list), np.array(X_list), np.array(I_list), np.array(Ra_list)]
    return  model_state_variable, np.array(meas_list), np.array(true_G_list)

