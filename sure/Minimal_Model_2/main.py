# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: The main function to run the model

from Initialization import total_t_list, N_meas, init_cond, Gb, Ib, meal_params, param_list, T, T_list, N
import numpy as np
import matplotlib.pyplot as plt
from Path_whole import generate_path_whole
from Noise_generator import noise_path
from Measurement import Plot_measurement
from FDA import FDA_control, FDA_param, FDA_filter
from optimization_all import optimize_filter_control
from optimization_sep import optimize_filter, optimize_param, optimize_control
from Cost import cost_computation

Filter = [0.01, 0.01, 0.01, 0.01]

h1 = 15;
h2 = 3;
h3 = 0.1;
h4 = 0.15
control_gain = [h1, h2, h3, h4]

##simulation with control
total_noise = noise_path(init_cond, N * N_meas, seed_num=1)
model_state_variable, noisy_meas, true_G = generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas,
                                              T, T_list, N, meal_params)

grad = FDA_filter(model_state_variable, true_G, init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)
print(grad)

cost = cost_computation(true_G, model_state_variable, Gb, Ib, control_gain)
n = 10

# Filter, cost_l, grad_l= optimize_control(init_cond, param_list, control_gain, Filter, total_noise,
#                                                                Gb, Ib, N_meas, T, T_list, N, meal_params,
#                                                                which='RMSprop', alpha=1e-4, momentum=0, beta=0.9, n=n)

Filter, control_gain, cost_l, grad_l, filter_l, control_l = optimize_filter_control(init_cond, param_list, control_gain, Filter, total_noise,
                                                               Gb, Ib, N_meas, T, T_list, N, meal_params,
                                                               which='RMSprop', alpha=1e-4, momentum=0, betas=0.9, n=n)
x = [i for i in range(n + 1)]
plt.plot(x, cost_l)
plt.xlabel("number of optimization steps")
plt.ylabel("cost")
plt.show()
