# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: The main function to run the model

from Initialization import tk_list, qk_list, tau, t_list, c1, c2, dt, I0, meas_time, p1, p2, p3, Gb, Ib, G0,X0, Ra_0, h, lam, end_time
import Plot, Glucose
import numpy as np
import Simulation, Measurement
import matplotlib.pyplot as plt
import Cost
import Insulin_ODE_Solver
##Some test


## Some simple control on insulin injection; later will be replaced by control from SGD
vn_list = np.zeros([1, len(t_list)])[0]
#print("No control *****************************************************")
#G_no_c, X_no_c, I_no_c = Simulation.path_generator(vn_list, p1, p2, p3, Gb, Ib, I0, c1, c2, meas_time, t_list, tk_list, qk_list, tau, G0, Ra_0, dt, h)

vn_list[100:280] = 240
#vn_list[70:80] = 20
#vn_list[160:170] = 20

print("Discrete **")

G_list, X_list, I_from_DOE_solver = Simulation.path_generator(vn_list, p1, p2, p3, Gb, Ib, I0, c1, c2, meas_time, t_list, tk_list, qk_list, tau, G0,X0, Ra_0, dt, h)
I_list = Insulin_ODE_Solver.insulin_solver(I0, t_list, vn_list, c1, c2)
print(Cost.cost_computation(G_list, Gb, lam, vn_list, end_time))
print("Continuous **")
G,X,I = Simulation.cont_path_generator(vn_list, p1, p2, p3, Gb, Ib, I0, c1, c2, t_list, tk_list, qk_list, tau, G0, X0, Ra_0)

plt.plot(t_list , G_list, label="with control")
#plt.plot(t_list[70:280], G_no_c[70:280], '--',label="no control")
plt.plot(t_list, G,  '--',   label="continuous simulation")
plt.title("G")
plt.legend()
plt.show()

plt.plot(t_list , X_list, label="with control")
#plt.plot(t_list[70:280], G_no_c[70:280], '--',label="no control")
plt.plot(t_list, X,  '--',   label="continuous simulation")
plt.title("X")
plt.legend()
plt.show()



plt.plot(t_list,I, label = "continuous solution")
#plt.plot(t_list[90:120],I_from_DOE_solver[90:120], label ="I from ODE")
plt.plot(t_list,I_list,'--',label ="discrete solution")
plt.title("I")
plt.legend()
plt.show()
noisy_meas = Measurement.measurement_generation( t_list, tk_list, qk_list, dt)
Measurement.Plot_measurement(noisy_meas, t_list, dt)