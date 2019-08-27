# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Unit control is completed here
from optimization import optimize

def advance_controller(init_cond, param_list, vn_list, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params):
    """
    take the noisy input, generate the control (insulin rate for the next T time),
    update the state and parameter estimates and advance the control model time T
    """

    #### SGD to get sub_vn_list, new parameter for this control period (Ziyu)

    param_list, cost_l, grad_l = optimize(init_cond, param_list, vn_list, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params, \
                                      which='RMSprop', alpha=1e-5, momentum=0, beta=0.9, n=200)

    #### generate the model process again using updated parameters  (Xinyu)




