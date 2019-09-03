# Author: Xinyu Li, Ziyu Lu
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: example to generate a path and plot

from Path_whole import generate_path_whole
from Initialization import init_cond, N, N_meas, param_list, Gb, Ib, T, T_list, meal_params, total_t_list
from Noise_generator import noise_path
from plotter import multi_plot
import matplotlib.pyplot as plt
import numpy as np

all_data = np.load('out50_lr2_w1000.npz')
print(sorted(all_data.files))
cost_l = all_data['cost_l']
filter_l = all_data['filter_l']
control_l = all_data['control_l']
gradF_l = all_data['gradF_l']
gradC_l = all_data['gradC_l']

algo = 'RMSprop'  # which optimization algorithm to use (Adam, RMSprop, or SGD)
alpha = 1e-2      # learning rate
momentum = 0      # momentum for SGD
beta1 = 0.9       # smoothing constant 1 (the only beta for RMSprop)
beta2 = 0.99      # smoothing constant 2 (the additional beta for Adam)
M = 1             # mini-batch size
n = 50          # number of gradient descent iterations

## Plot the change in the cost, filter, control, and gradients
multi_plot(cost_l, p=2, which=algo, nIter=n, alpha=alpha, M=M, beta1=beta1, beta2=beta2, log=True, label=None)
multi_plot(filter_l, p=1, which=algo, nIter=n, alpha=alpha, M=M, beta1=beta1, beta2=beta2, log=False, label='K')
multi_plot(control_l, p=1, which=algo, nIter=n, alpha=alpha, M=M, beta1=beta1, beta2=beta2, log=False, label='H')
# multi_plot(gradF_l[0], p=2, which=algo, nIter=n, alpha=alpha, M=M, beta1=beta1, beta2=beta2, log=False, label='grad K')
multi_plot(gradF_l, p=1, which=algo, nIter=n, alpha=alpha, M=M, beta1=beta1, beta2=beta2, log=False, label='grad K')

# multi_plot(gradC_l, p=1, which=algo, nIter=n, alpha=alpha, M=M, beta1=beta1, beta2=beta2, log=False, label='grad H')

## Compare the results before and after training
Filter1 = all_data['Filter']
print(Filter1)
# Filter1 = filter_l[:,-1]
# Filter1 = [0.01, 0.01, 0.01, 0.01]
control_gain1 = all_data['control_gain']
print(control_gain1)
# control_gain1 = control_l[:,-1]
total_noise = noise_path(init_cond, N * N_meas)
model_state_variable1, Z1, true_state_variable1 = generate_path_whole(init_cond, param_list, control_gain1, Filter1,
                                                                   total_noise, Gb, Ib, N_meas, T, T_list, N,
                                                                   meal_params)
Filter2 = [0.1, 0.01, 0.01, 0.37]
control_gain2 = np.array([15, 3, 0.1, 0.5])*0.001
model_state_variable2, Z2, true_state_variable2 = generate_path_whole(init_cond, param_list, control_gain2, Filter2,
                                                                   total_noise, Gb, Ib, N_meas, T, T_list, N,
                                                                   meal_params)

## Plot G
plt.plot(total_t_list, model_state_variable1[0], label="model estimation after training")
plt.plot(total_t_list, true_state_variable2[0], label="in-silico subject with initial control")
plt.plot(total_t_list, true_state_variable1[0], label="in-silico subject with trained control")



# plt.plot(total_t_list, len(total_t_list)*[140], label="upper bound")
# plt.plot(total_t_list, len(total_t_list)*[80], label="lower bound")
plt.ylabel("G")
plt.xlabel("t")
plt.title("Glucose $G(t)$")
plt.legend()
plt.show()

## Plot X
plt.plot(total_t_list, model_state_variable1[1], label="model estimation after training")
plt.plot(total_t_list, true_state_variable1[1], label="in-silico subject after training")
plt.plot(total_t_list, true_state_variable2[1], label="in-silico subject before training")


plt.ylabel("X")
plt.xlabel("t")
plt.title("Remote insulin $X(t)$")
plt.legend()
plt.show()

## Plot I
plt.plot(total_t_list, model_state_variable1[2], label="model estimation after training")
plt.plot(total_t_list, true_state_variable1[2], label="in-silico subject after training")
plt.plot(total_t_list, true_state_variable2[2], label="in-silico subject before training")


plt.ylabel("I")
plt.xlabel("t")
plt.title("Insulin $I(t)$")
plt.legend()
plt.show()

## Plot Ra
plt.plot(total_t_list, model_state_variable1[3], label="model estimation after training")
plt.plot(total_t_list, true_state_variable1[3], label="in-silico subject after training")
plt.plot(total_t_list, true_state_variable2[3], label="in-silico subject before training")


plt.ylabel("Ra")
plt.xlabel("t")
plt.title("Glucose $Ra(t)$")
plt.legend()
plt.show()

