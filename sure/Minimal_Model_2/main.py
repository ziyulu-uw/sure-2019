# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: The main function to run the model

from Initialization import tk_list, qk_list,t_list, dt, meas_time,init_cond,  Gb, Ib, h, lam, end_time, meal_time, param_list
import numpy as np
import Measurement
import matplotlib.pyplot as plt
from Cost import cost_computation
from Simulation import path_generator
from FinteDiffApprox import FDA
from optimization import optimize


## a no control simulation
v0_list = np.zeros([1, len(t_list)])[0]   # a control list which is zero
G_no_c, X_no_c, I_no_c, Ra_no_c = path_generator(init_cond, param_list, v0_list, Gb, Ib, meas_time, t_list, tk_list, qk_list, meal_time, dt, h)
## Some simple control on insulin injection; later will be replaced by control from SGD
vn_list = np.zeros([1, len(t_list)])[0]
vn_list[100:280] = 240
vn_list[600:1000] = 300
vn_list[1400:1700] = 120

#simulation with control
# G, X, I, Ra = path_generator(init_cond, param_list,vn_list, Gb, Ib, meas_time, t_list, tk_list, qk_list, meal_time, dt, h)
# compare with control and no control senario
# plt.plot(t_list , G, label="with control")
# plt.plot(t_list, G_no_c, '--',label="no control")
# plt.title("Glucose $G(t)$")
# plt.xlabel("min")
# plt.ylabel("$G$ (mg/dl)")
# plt.legend()
# plt.show()
#
# plt.plot(t_list , X, label="with control")
# plt.plot(t_list, X_no_c, '--',label="no control")
# plt.title("$X(t)$")
# plt.xlabel("min")
# plt.ylabel("$X$ (mg/dl)")
# plt.legend()
# plt.show()
#
# plt.plot(t_list,I, label ="with control")
# plt.plot(t_list,I_no_c,'--',label ="no control")
# plt.title("Insulin $I(t)$")
# plt.xlabel("min")
# plt.ylabel("$I$ (mU/l)")
# plt.legend()
# plt.show()
#
# plt.plot(t_list,Ra, label ="with control")
# plt.plot(t_list,Ra_no_c,'--',label ="no control")
# plt.title("Glucose Rate of Appearance $R_a$")
# plt.xlabel("min")
# plt.ylabel("$R_a$ (mg/kg/min)")
# plt.legend()
# plt.show()
#
# noisy_meas = Measurement.measurement_generation(vn_list, t_list,  dt,h, meas_time)
# Measurement.Plot_measurement(noisy_meas, t_list, dt)

# grad = FDA(init_cond, param_list,vn_list, Gb, Ib, meas_time, t_list, tk_list, qk_list, meal_time, dt, h)
# print(grad)

n = 200
param_list, cost_l, grad_l = optimize(init_cond, param_list,vn_list, Gb, Ib, meas_time, t_list, tk_list, qk_list, meal_time, dt, h,\
                                      which='RMSprop', alpha=1e-5, momentum=0, beta=0.9, n=n)

x = [i for i in range(n+1)]
plt.plot(x, cost_l)
# plt.title("Glucose Rate of Appearance $R_a$")
plt.xlabel("number of optimization steps")
plt.ylabel("cost")
plt.show()
