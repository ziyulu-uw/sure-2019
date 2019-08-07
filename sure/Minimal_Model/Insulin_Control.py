# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Administered Insulin is controlled here

import scipy
import matplotlib.pyplot as plt
import numpy as np
from DE_solver import G_X_sys                      # glucose minimal model is solved here
from Insulin_Input import phar_kin   # insulin function is simulated here
from Cost import cost_computation
from Parameters import glucose, insulin, time_param, cost


def simulation_wz_control(u, r):
    """simulate with a control exerted on insulin input vB
    vB = vB*u
    :param  u - control
    :return total cost of the process"""
    ## Plot I(t) given by pharmacokinetics model in paper 3
    I_t = phar_kin.solveODE(u, Plot=False)

    ## Plot G(t) simulated by minimal model
    G, X = G_X_sys(u, 0, Plot=False)     # G_X_sys(control, index_to_choose_model, Plot)
    total_cost = cost_computation(I_t, phar_kin.Ib, G, glucose.Gb, r)
    return total_cost

def find_optimal_control(r):
    #def find_optimal_control():
    u_list =  np.linspace(0,2, 11)
    cost_list = []
    for u in u_list:
        cost_list.append(simulation_wz_control(u, r))

    # find the index of minimal cost in cost_list
    idx1 = cost_list.index(min(cost_list))
    u_temp_min   = u_list[idx1]  # the optimal u in the u_list

    # find the reduced scope for find optimal control
    u1 = u_temp_min-0.05         # notice: here value that control the scope need to be revised if r has changed\
    if u1<0:                    # one can decide the size of searching scope given the status scipy.otimize result.\
        u1 = 0                  # (tip: reduce the scope when r is small)
    u2 = u_temp_min+0.05
    bdd = (u1, u2)              # a reduced scope for find the optimal control

    sol = scipy.optimize.minimize(simulation_wz_control, u_temp_min, bounds=(bdd,), args=(r,))
    print(sol)
    plt.xlabel("control gain u")
    plt.ylabel("total cost")
    plt.plot(u_list, cost_list)
    plt.title("total cost w.r.t control gain u with r = {:.1e}".format(r))
    plt.plot([sol.x[0]], [sol.fun], 'o', marker='o', markersize=3, color="red", label="optimal u=%.2f, cost=%.2f"%(sol.x[0], sol.fun))
    plt.legend()
    plt.grid()
    plt.show()

    return sol.x[0], sol.fun
