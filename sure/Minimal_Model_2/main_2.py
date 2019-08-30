# Author: Xinyu Li, Ziyu Lu
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: example to generate a path and plot

from Path_whole import generate_path_whole
from Initialization import init_cond, N, N_meas, param_list, Gb, Ib, T, T_list, meal_params, total_t_list
from Noise_generator import noise_path
import matplotlib.pyplot as plt
import numpy as np

all_data = np.load('out.npz')
Filter1 = all_data['Filter']
control_gain1 = all_data['control_gain']
total_noise = noise_path(init_cond, N * N_meas)
model_state_variable1, Z1, true_state_variable1 = generate_path_whole(init_cond, param_list, control_gain1, Filter1,
                                                                   total_noise, Gb, Ib, N_meas, T, T_list, N,
                                                                   meal_params)
Filter2 = [0.01, 0.01, 0.01, 0.01]
control_gain2 = [15, 3, 0.1, 0.5]
model_state_variable2, Z2, true_state_variable2 = generate_path_whole(init_cond, param_list, control_gain2, Filter2,
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

