# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Some parameters and inputs are initialized here

import numpy as np

# Meal Input
# meal time
tk_list = np.array([60, 350, 720])*0  # unit: min
# the time that the subject take the meal
meal_time = 25
# meal intake value
qk_list = np.array([3000, 4500, 3500])*0 / meal_time  # unit: mg/min, from ref 1 in Xinyu's writeup
# wrap all the meal related variables into one variable
meal_params = (tk_list,qk_list,meal_time)

# digestion coefficient
tau =  100  # unit: 1  [unknown parameter!!!]
# initial glucose rate of appearance
Ra_0 = 5  # unit: mg/kg/min

# Insulin Parameter
c1 = 0.25  # unit: min^-1   [unknown parameter!!!]
c2 = 0.2  # unit: min^-1   [unknown parameter!!!]
I0 = 0.38  # unit: mU/l     [unknown initial condition]
Ib = 0

# Minimal Model Parameter
p1 = 3.6  # param determined by bio-experiments (unit: min^-1)
p2 = 0.0122  # unit: min^-1
p3 = 1.7e-5  # unit: min^-2 mU/l
Gb = 125  # basal plasma glucose (unit:mg/dl)
G0 = 130
X0 = 0
"""A fasting blood sugar level less than 100 mg/dL (5.6 mmol/L) is normal.
 A fasting blood sugar level from 100 to 125 mg/dL (5.6 to 6.9 mmol/L) is considered prediabetes.
  If it's 126 mg/dL (7 mmol/L) or higher on two separate tests, you have diabete"""

# Wrap the parameters into a list:
param_list = [p1, p2, p3, tau, c1, c2]
init_cond  = [G0, X0, I0, Ra_0]
# Cost function parameter
lam = 0.5

# Code Parameter
end_time  = 1000            # unit: min
T         = 1000             # unit: min;    Time for one control period (Better to choose a number that divides end_time)
N         = end_time//T     # we will call the controller N times in a complete simulation NT = end_time

# time discretization for measurements in one control period
dt        = 5                             # unit: min;    Time interval between two measurements
N_meas    = int(T / dt)                   # num of measurements in ine control period
T_list    = np.linspace(0, T, N_meas+1)   # time discretization in one control period
meas_time = np.linspace(0, T-dt, N_meas)  # time when measurements happen in one control period

N_total   = N_meas*N                             # total number of measurements in the whole run
total_t_list    = np.linspace(0, end_time, N_total+1)  # time discretization in the whole run
