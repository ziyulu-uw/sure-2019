# Author: Xinyu Li, Ziyu Lu
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: example to generate a path and plot

from Path_whole import generate_path_whole
from Initialization import init_cond, N, N_meas, param_list, Gb, Ib, T, T_list, meal_params, total_t_list
from Noise_generator import noise_path
import matplotlib.pyplot as plt
import numpy as np


all_data = np.load('out100_lr4_b9.npz')
Filter = all_data['Filter']
control_gain = all_data['control_gain']
total_noise = noise_path(init_cond, N * N_meas)
model_state_variable1, Z1, true_state_variable1 = generate_path_whole(init_cond, param_list, control_gain, Filter,
                                                                   total_noise, Gb, Ib, N_meas, T, T_list, N,
                                                                   meal_params)
Filter = [0.01, 0.01, 0.01, 0.01]
control_gain = [15, 3, 0.1, 0.5]
total_noise = noise_path(init_cond, N * N_meas)
model_state_variable2, Z2, true_state_variable2 = generate_path_whole(init_cond, param_list, control_gain, Filter,
                                                                   total_noise, Gb, Ib, N_meas, T, T_list, N,
                                                                   meal_params)

## Plot G
plt.plot(total_t_list, model_state_variable1[0], label="model estimation after training")
plt.plot(total_t_list, true_state_variable2[0], label="in-silico subject before")
plt.plot(total_t_list, true_state_variable1[0], label="in-silico subject after")

# plt.plot(total_t_list, len(total_t_list)*[140], label="upper bound")
# plt.plot(total_t_list, len(total_t_list)*[80], label="lower bound")
plt.ylabel("G")
plt.xlabel("t")
plt.title("Glucose $G(t)$")
plt.legend()
plt.show()

## Plot Ra
plt.plot(total_t_list, model_state_variable1[3], label="model estimation after training")
plt.plot(total_t_list, true_state_variable2[3], label="in-silico subject before")
plt.plot(total_t_list, true_state_variable1[3], label="in-silico subject after")

plt.ylabel("Ra")
plt.xlabel("t")
plt.title("Glucose $Ra(t)$")
plt.legend()
plt.show()


'''
from wrapper import optim_wrapper
from plotter import multi_plot
Filter = [0.01, 0.01, 0.01, 0.01]
control_gain = [15, 3, 0.1, 0.5]
control_l, cost_l, grad_l = optim_wrapper(init_cond, param_list, control_gain, Filter, Gb, Ib, N_meas, T, T_list, N,
            meal_params, which='RMSprop', alpha=1e-3, momentum=0, beta1=0.9, beta2=0.99, M=1, n=10, fname=None)
multi_plot(cost_l, p=2, which='RMSprop', nIter=1000, alpha=1e-5, M=1, betas=0.99, log=True, label=None)
multi_plot(control_l, p=1, which='RMSprop', nIter=1000, alpha=1e-5, M=1, betas=0.99, log=False, label='H')
multi_plot(grad_l, p=1, which='RMSprop', nIter=1000, alpha=1e-5, M=1, betas=0.99, log=False, label='grad H')
'''
