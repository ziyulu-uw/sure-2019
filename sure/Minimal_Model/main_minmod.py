# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: Run this program to see the simulation of minimal model with different I(t) model

from DE_solver import G_X_sys                      # glucose minimal model is solved here
from Insulin_Input import phar_kin, insulin_data   # insulin function is simulated here
from Cost import cost_computation
from Parameters import glucose, insulin, time_param, cost
from Insulin_Control import find_optimal_control

"""Please choose the insulin model I(t)
set idx to 0: pharmacokinetics model for I(t)
set idx to 1: linear approximation from insulin data """

## Control Gain
u   = 1   # when u=1, there is no control on vB (Administered Insulin Input)

## Plot I(t) given by pharmacokinetics model in paper 3
idx = 0
if idx == 0:
    I_t = phar_kin.solveODE(u, Plot=True)
    #Ib  = I_t[-1][-1]   # set basal value of the insulin as the Insulin model at the end time
    phar_kin.plot_v()
    ## Plot G(t) simulated by minimal model
    G, X = G_X_sys(u, idx, Plot=True)
    total_cost = cost_computation(I_t, phar_kin.Ib, G, glucose.Gb, cost.r)
    print("total cost with default vB=%d:             %.2f"%(phar_kin.vB,total_cost))

## Plot I(t) given by insulin data in paper 1
idx  = 2
if idx == 1:
    I_t, dI = insulin_data.linear_approx_eqn(insulin_data.t, Plot=True)
    #Ib  = I_t[-1]       # set basal value of the insulin as the Insulin model at the end time
    ## Plot G(t) simulated by minimal model
    G, X = G_X_sys(u, idx, Plot=True)
    total_cost = cost_computation(I_t, phar_kin.Ib, G, glucose.Gb, cost.r)
    print("total cost with given Insulin Data: %.2f"%total_cost)

## Find the optimal Control and cost
r = 5e-3
print("        r:",r)
find_optimal_control(r)  # you need to manually change the scope of bdd in scipy.minimize to have some case success