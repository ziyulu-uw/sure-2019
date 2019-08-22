# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: The main function to run the model

from Initialization import tk_list, qk_list, tau, t_list, c1, c2, dt, I0, meas_time, p1, p2, p3, Gb, Ib, G0,X0, Ra_0, h, lam, end_time, meal_time
import numpy as np
import Measurement
import matplotlib.pyplot as plt
from Cost import cost_computation
from Simulation import path_generator

## Some simple control on insulin injection; later will be replaced by control from SGD
vn_list = np.zeros([1, len(t_list)])[0]

# a no control simulation
G_no_c, X_no_c, I_no_c, Ra_no_c = path_generator(vn_list, p1, p2, p3, Gb, Ib, I0, c1, c2, meas_time, t_list, tau, G0,X0, Ra_0, dt, h, tk_list, qk_list, meal_time)

vn_list[100:280] = 240
vn_list[600:1000] = 300
vn_list[1400:1700] = 120


G, X, I, Ra = path_generator(vn_list, p1, p2, p3, Gb, Ib, I0, c1, c2, meas_time, t_list, tau, G0,X0, Ra_0, dt, h, tk_list, qk_list, meal_time)

# find out the cost
print("cost with control:", cost_computation(G, Gb, lam, vn_list, end_time))

# compare with control and no control senario
plt.plot(t_list , G, label="with control")
plt.plot(t_list, G_no_c, '--',label="no control")
plt.title("Glucose $G(t)$")
plt.xlabel("min")
plt.ylabel("$G$ (mg/dl)")
plt.legend()
plt.show()

plt.plot(t_list , X, label="with control")
plt.plot(t_list, X_no_c, '--',label="no control")
plt.title("$X(t)$")
plt.xlabel("min")
plt.ylabel("$X$ (mg/dl)")
plt.legend()
plt.show()

plt.plot(t_list,I, label ="with control")
plt.plot(t_list,I_no_c,'--',label ="no control")
plt.title("Insulin $I(t)$")
plt.xlabel("min")
plt.ylabel("$I$ (mU/l)")
plt.legend()
plt.show()

plt.plot(t_list,Ra, label ="with control")
plt.plot(t_list,Ra_no_c,'--',label ="no control")
plt.title("Glucose Rate of Appearance $R_a$")
plt.xlabel("min")
plt.ylabel("$R_a$ (mg/kg/min)")
plt.legend()
plt.show()

noisy_meas = Measurement.measurement_generation(vn_list, t_list,  dt,h, meas_time)
Measurement.Plot_measurement(noisy_meas, t_list, dt)