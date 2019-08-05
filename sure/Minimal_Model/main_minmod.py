# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: Run this program to see the simulation of minimal model
from DE_solver import G_X_sys
from Insulin_Input import phar_kin, insulin_data

"""Please choose the insulin model I(t)
set idx to 0: pharmacokinetics model for I(t)
set idx to 1: linear approximation from insulin data """
idx = 1

if idx == 0:
    ## Plot I(t) given by pharmacokinetics model in paper 3
    I_t = phar_kin.solveODE(Plot=True)
    Ib  = I_t[-1][-1]   # set basal value of the insulin as the Insulin model at the end time

if idx == 1:
    I_t, dI = insulin_data.linear_approx_eqn(insulin_data.t, Plot=True)
    Ib  = I_t[-1]       # set basal value of the insulin as the Insulin model at the end time

## Plot G(t) simulated by minimal model
G, X = G_X_sys(idx, Plot=True)