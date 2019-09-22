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
import numpy as np
#Filter = np.array([0.76, -0.00000000, 0.000000, 0.56]) # initial filter gain
#Filter = np.array([0.7, -1.7, 0, 2.8])
Filter = np.array([ 0.73336523, -0.12375505, 17.76651593 , 2.80219249])*0
#control_gain = np.array([1.3, 2, 0.9, 0.03])  # initial control gain
control_gain = np.array([68.32140527, -1.78250341, -0.47587041, 21.57038705])
## Compare the results before and after training
total_noise = noise_path(init_cond, N * N_meas, var1=0, var2=0)
state_variable, Z1, true_state_variable1 = generate_path_whole(init_cond, param_list, control_gain, Filter,
                                                                   total_noise, Gb, Ib, N_meas, T, T_list, N,
                                                                   meal_params)

# grad, cost = FDA_filter(state_variable, true_state_variable1, init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N,
#                meal_params)

# true_G = true_state_variable1[0]
# grad, cost = FDA_filter(state_variable, true_state_variable1, init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list,
#            N, meal_params)
#
# algo = 'RMSprop'  # which optimization algorithm to use (Adam, RMSprop, or SGD)
# alpha = 1e-3      # learning rate
# momentum = 0      # momentum for SGD
# beta1 = 0.9       # smoothing constant 1 (the only beta for RMSprop)
# beta2 = 0.99      # smoothing constant 2 (the additional beta for Adam)
# M = 1             # mini-batch size
# n = 10            # number of gradient descent iterations
#
# Filter, control_gain, cost_l, filter_l, control_l, gradF_l, gradC_l = optim_wrapper(init_cond, param_list, control_gain, Filter, Gb, Ib, N_meas, T, T_list, N,
#             meal_params, which=algo, alpha=alpha, momentum=momentum, beta1=beta1, beta2=beta2, M=M, n=n, fname='out')

G,X,I,Ra = state_variable

plt.plot(total_t_list, G,label="model glucose")
plt.plot(total_t_list, true_state_variable1[0],label = "true glucose")
plt.legend()
plt.title("G")
plt.show()

plt.plot(total_t_list, I,label="model insulin")
plt.plot(total_t_list, true_state_variable1[2],label="true insulin")
plt.legend()
plt.title("I")
plt.show()

plt.plot(total_t_list, X,label="model remote insulin")
plt.plot(total_t_list, true_state_variable1[1],label="true remote insulin")
plt.legend()
plt.title("X")
plt.show()

plt.plot(total_t_list, Ra,label="model Ra")
plt.plot(total_t_list, true_state_variable1[-1],label="true Ra")
plt.legend()
plt.title("Ra")
plt.show()


h1,h2,h3,h4 = control_gain
vn = h1 * (G - Gb) + h2 * X + h3 * (I - Ib) + h4 * Ra

true_G, true_X, true_I, true_Ra = true_state_variable1
true_vn = h1 * (true_G - Gb) + h2 * true_X + h3 * (true_I - Ib) + h4 * true_Ra
plt.plot(vn,label="model vn")
plt.plot(true_vn,label="true vn")

plt.title("vn")
plt.legend()
plt.show()