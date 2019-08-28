# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: The main function to run the model

from Initialization import total_t_list, N_meas, init_cond,  Gb, Ib, meal_params, param_list, T, T_list, N
import numpy as np
import matplotlib.pyplot as plt
from Path_whole import generate_path_whole
from Noise_generator import noise_path
from Measurement import Plot_measurement
from FinteDiffApprox import FDA_control
from optimization import optimize_control
from Cost import cost_computation
## Some simple control on insulin injection; later will be replaced by control from SGD
vn_list = np.arange(N_meas*N)
Filter = [0.01,1e-9,0.01,0.01]

h1 = 15; h2 = 1e-6; h3 = 0.1; h4 = 0.15
control_gain=[h1, h2, h3, h4]

##simulation with control
total_noise = noise_path(init_cond, N*N_meas, seed_num=1)

#G,X,I,Ra,noisy_meas = generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)
'''
##Plot G X I Ra
plt.plot(total_t_list , G)
plt.title("Glucose $G(t)$")
plt.xlabel("min")
plt.ylabel("$G$ (mg/dl)")
#plt.legend()
plt.show()

plt.plot(total_t_list , X)
plt.title("$X(t)$")
plt.xlabel("min")
plt.ylabel("$X$ (mg/dl)")
#plt.legend()
plt.show()

plt.plot(total_t_list, I)
plt.title("Insulin $I(t)$")
plt.xlabel("min")
plt.ylabel("$I$ (mU/l)")
#plt.legend()
plt.show()

plt.plot(total_t_list, Ra)
plt.title("Glucose Rate of Appearance $R_a$")
plt.xlabel("min")
plt.ylabel("$R_a$ (mg/kg/min)")
#plt.legend()
plt.show()

Plot_measurement(noisy_meas, total_t_list)
'''

#grad =FDA_control(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)
#print(grad)
#cost = cost_computation(G, X, I, Ra, Gb, Ib, control_gain)
#print(cost)

n = 30
control_gain, cost_l, grad_l = optimize_control(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params,
                 which='RMSprop', alpha=1e-5, momentum=0, beta=0.9, n=n)
x = [i for i in range(n+1)]
plt.plot(x, cost_l)
plt.xlabel("number of optimization steps")
plt.ylabel("cost")
plt.show()
