# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: September 2019
# Description: a temporary module to test


import matplotlib.pyplot as plt
from FDA import FDA_filter
from Initialization import init_cond, param_list, Gb, Ib, N_meas, T, T_list, N, meal_params, total_t_list
from Noise_generator import noise_path
from Path_whole import generate_path_whole
from wrapper import optim_wrapper

Filter = [0, 0.01, 0.01, 0.1]  # initial filter gain
control_gain = [15, 3, 0.1, 0.5]   # initial control gain


## Compare the results before and after training
total_noise = noise_path(init_cond, N * N_meas)
state_variable, Z1, true_state_variable1 = generate_path_whole(init_cond, param_list, control_gain, Filter,
                                                                   total_noise, Gb, Ib, N_meas, T, T_list, N,
                                                                   meal_params)
true_G = true_state_variable1[0]
grad, cost = FDA_filter(state_variable, true_state_variable1, init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list,
           N, meal_params)

algo = 'RMSprop'  # which optimization algorithm to use (Adam, RMSprop, or SGD)
alpha = 1e-3      # learning rate
momentum = 0      # momentum for SGD
beta1 = 0.9       # smoothing constant 1 (the only beta for RMSprop)
beta2 = 0.99      # smoothing constant 2 (the additional beta for Adam)
M = 1             # mini-batch size
n = 10            # number of gradient descent iterations

Filter, control_gain, cost_l, filter_l, control_l, gradF_l, gradC_l = optim_wrapper(init_cond, param_list, control_gain, Filter, Gb, Ib, N_meas, T, T_list, N,
            meal_params, which=algo, alpha=alpha, momentum=momentum, beta1=beta1, beta2=beta2, M=M, n=n, fname='out')

G,X,I,Ra = state_variable
plt.plot(total_t_list, Ra)
plt.plot(total_t_list, true_state_variable1[-1])
plt.show()

plt.plot(total_t_list, G)
plt.plot(total_t_list, true_state_variable1[0])
plt.show()