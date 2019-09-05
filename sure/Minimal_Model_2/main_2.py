from wrapper import optim_wrapper
from plotter import multi_plot
from Initialization import init_cond, N, N_meas, param_list, Gb, Ib, T, T_list, meal_params, total_t_list
from Noise_generator import noise_path
from Path_whole import generate_path_whole
import matplotlib.pyplot as plt
import numpy as np

Filter = [1, 0.01, 0.01, 0.37]  # initial filter gain
control_gain = np.array([15, 3, 0.1, 0.5])*0.001   # initial control gain
algo = 'RMSprop'  # which optimization algorithm to use (Adam, RMSprop, or SGD)
alpha = 1e-2      # learning rate
momentum = 0      # momentum for SGD
beta1 = 0.9       # smoothing constant 1 (the only beta for RMSprop)
beta2 = 0.99      # smoothing constant 2 (the additional beta for Adam)
M = 1             # mini-batch size
n = 10            # number of gradient descent iterations

Filter, control_gain, cost_l, filter_l, control_l, gradF_l, gradC_l = optim_wrapper(init_cond, param_list, control_gain, Filter, Gb, Ib, N_meas, T, T_list, N,
            meal_params, which=algo, alpha=alpha, momentum=momentum, beta1=beta1, beta2=beta2, M=M, n=n, fname='out')

## Plot the change in the cost, filter, control, and gradients
multi_plot(cost_l, p=2, which=algo, nIter=n, alpha=alpha, M=M, beta1=beta1, beta2=beta2, log=True, label=None)
multi_plot(filter_l, p=1, which=algo, nIter=n, alpha=alpha, M=M, beta1=beta1, beta2=beta2, log=False, label='K')
multi_plot(control_l, p=1, which=algo, nIter=n, alpha=alpha, M=M, beta1=beta1, beta2=beta2, log=False, label='H')
multi_plot(gradF_l, p=1, which=algo, nIter=n, alpha=alpha, M=M, beta1=beta1, beta2=beta2, log=False, label='grad K')
multi_plot(gradC_l, p=1, which=algo, nIter=n, alpha=alpha, M=M, beta1=beta1, beta2=beta2, log=False, label='grad H')

## Compare the results before and after training
total_noise = noise_path(init_cond, N * N_meas)
model_state_variable1, Z1, true_state_variable1 = generate_path_whole(init_cond, param_list, control_gain, Filter,
                                                                   total_noise, Gb, Ib, N_meas, T, T_list, N,
                                                                   meal_params)
Filter2 = [0.01, 0.01, 0.01, 0.01]
control_gain2 = [15, 3, 0.1, 0.5]
model_state_variable2, Z2, true_state_variable2 = generate_path_whole(init_cond, param_list, control_gain2, Filter2,
                                                                   total_noise, Gb, Ib, N_meas, T, T_list, N,
                                                                   meal_params)

## Plot G
# plt.plot(total_t_list, model_state_variable1[0], label="model estimation after training")
#plt.plot(total_t_list, true_state_variable2[0], label="in-silico subject with initial control")
plt.plot(total_t_list, true_state_variable1[0], label="in-silico subject with trained control")
plt.plot(total_t_list, model_state_variable1[0], label="model estimation")

# plt.plot(total_t_list, len(total_t_list)*[140], label="upper bound")
# plt.plot(total_t_list, len(total_t_list)*[80], label="lower bound")
plt.ylabel("G")
plt.xlabel("t")
plt.title("Glucose $G(t)$")
plt.legend()
plt.show()

## Plot Ra
plt.plot(total_t_list, model_state_variable1[3], label="model estimation after training")
#plt.plot(total_t_list, true_state_variable2[3], label="in-silico subject with initial control")
plt.plot(total_t_list, true_state_variable1[3], label="in-silico subject with trained control")

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