# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This program generate the path in one control period \
#              Decide the control in one control period

from ODE_solver import Minimod_ODE_solver
import numpy as np

def generate_path_unit(init_cond, param_list, sub_vn_list, Filter, Z, noise, Gb, Ib, sim_idx, N_meas, T, T_list, meal_params, idx):
    """
    this function generates the path in [one control period]
    Take the noisy input, the control, and filter as input -> update the states in this period

    Notice: instead of the whole path, it will generate a path of length T, (NT = end_time)
    If the function is called for ODE model, set idx to be zero, so that no process noise will be added;
    If the function is called for building the person, set idx to be one. Process noise will be added to correct the model.

    @:param true_init_cond                   true initial condition used in making up the human subject
    @:param init_cond = G0,X0,I0,Ra_0        initial conditions of glucose level,
                                             remote insulin level, insulin level, glucose disappearance rate
    @:param param_list:                      [p1, p2, p3, tau, c1, c2]
    @:param vn_list:                         a list of controls exerted in the control period, dimension = len(T_list)
    @:param Filter = [K1,K2,K3,K4]           filter should be decided by SGD
    @:param Z:                               G measurements in this control period
    @:param noise = (process_noise, observation_noise)   4*N_meas, 1*N_meas
    @:param Gb, Ib:                          basal values [known parameter]
    @:param N_meas:                          the number of measurements in one control simulation
    @:param T_list:                          time discretization in one control simulation
    @:param T:                               Time for one control period
    @:param meal_params                      meal time and intake amount

    :return G,X,I,Ra                         arrays of state variables in one control period """

    G0, X0, I0, Ra_0 = init_cond
    G_list     = [G0]
    X_list     = [X0]
    I_list     = [I0]
    Ra_list    = [Ra_0]

    if idx==0:
        process_noise = np.zeros([4, N_meas])
    elif idx==1:
        process_noise = noise[0]
        Filter = [0,0,0,0]
    else:
        print("invalid index")
        return

    for i in range(N_meas):
        ## Discretize to sub section in one dt
        sub_t_list  =  T_list[int(i): int((i+1)+1)]

        ## Solve the ODE in the sub section
        state_variables =  Minimod_ODE_solver(init_cond, sub_t_list, param_list, sub_vn_list[i], Gb, Ib, sim_idx, T, meal_params)

        '''the process noise would be added only when the function is used for making up a human subject'''
        for j in range(len(state_variables)):
            # add process noise
            state_variables[j] += process_noise[j,i]
            # add the filter: Gn+1 = Gn+1^ + K(measurement - Gn+1^)
            state_variables[j] = state_variables[j] + Filter[j]*(Z[i]-state_variables[j])

        G,X,I,Ra = state_variables

        ## Use the last value in G,X,I,Ra as initial condition for the next iteration
        init_cond = state_variables.copy()

        ## Record
        G_list.append(G)
        X_list.append(X)
        I_list.append(I)
        Ra_list.append(Ra)

    return np.array(G_list), np.array(X_list), np.array(I_list), np.array(Ra_list)























