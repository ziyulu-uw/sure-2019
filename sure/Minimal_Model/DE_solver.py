# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program is a differential equation solver for minimal model

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from Parameters import time_param, glucose, insulin
from Insulin_Input import phar_kin, insulin_data
from Glucose_Rate import Ra

def GXI_system1(x, t, p1, p2, p3, Gb, Ib, sol_I):
    """glucose (G) - remote insulin (X) - plasma insulin (I) system
     :param x          - x = [G, X, I]
     :param p1, p2, p3 - bio parameters (unit: min^-1, min^-1, min^-1 mU/l)
     :param Gb, Ib     - basal plasma glucose (mmol/l) and insulin (mU/l)
     :param sol_I      - I(t) insulin approximation function data
     """
    G = x[0]
    X = x[1]

    dG_dt = -(p1 + X) * G + p1 * Gb + Ra(t)
    dX_dt = -p2 * X + p3 * (phar_kin.I(t, sol_I) - Ib)

    #debug
    #if t<75:
        #print(t,G,Gb,G-Gb)
        #print("t:",t,"-p1(G-Gb):",-p1*(G-Gb), " -GX:",-G*X, " G:",G, " dG:", dG_dt)
        #print("Ra:", Ra(t), "p1*Gb", p1*Gb, "-(p1+X)",-(p1 + X)  *G)
        #print(t, "dG:", dG_dt, "  G:",G, "  X:", X, "  dX_dt:",dX_dt)
    return [dG_dt, dX_dt]


def GXI_system2(x, t, p1, p2, p3, Gb, Ib):
    """glucose (G) - remote insulin (X) - plasma insulin (I) system
     :param x          - x = [G, X, I]
     :param p1, p2, p3 - bio parameters (unit: min^-1, min^-1, min^-1 mU/l)
     :param Gb, Ib     - basal plasma glucose (mmol/l) and insulin (mU/l)
     :param sol_I      - I(t) insulin approximation function data
     """
    G = x[0]
    X = x[1]

    dG_dt = -(p1 + X) * G + p1 * Gb + Ra(t)
    dX_dt = -p2 * X + p3 * (insulin_data.I_linear_data(t) - Ib)

    #debug
    #if t<75:
        #print("t:",t,"-p1(G-Gb):",-p1*(G-Gb), " Ra:",Ra(t), " G:",G, " dG:", dG_dt)
        #print("Ra:", Ra(t), "p1*Gb", p1*Gb, "-(p1+X)",-(p1 + X)  *G)
        #print(t, "dG:", dG_dt, "  G:",G, "  X:", X, "  dX_dt:",dX_dt)
    return [dG_dt, dX_dt]


def G_X_sys(Input_Index, Plot):
    """Solve the Differential equations for G(t) and X(t)
    :param Input_Index=0 if you want to use input function given by pharmacokinetics model
           Input_Index=1 if you want to use input function by linear approximation of insulin data"""

    ## Parameter Setting
    p1, p2, p3 = glucose.p1, insulin.p2, insulin.p3  # bio parameters
    Gb = glucose.Gb  # basal plasma glucose

    if Input_Index==0:
        Ib = phar_kin.Ib  # basal plasma insulin
        t  = np.linspace(0,phar_kin.t1-1,1000)


    if Input_Index==1:
        Ib = insulin_data.Ib
        t = np.linspace(0,insulin_data.t1-1,1000)

    # Time Linspace
    #t = np.linspace(0,199,1000)

    ## initial condition for [G, X]
    x0 = [glucose.G0, 0]

    ## solve the ODE system based on chosen Insulin model I(t)
    if Input_Index==0:
        sol_I  = phar_kin.solveODE(Plot=False)  # Get I(t) first, then use I(t) to solve the ODE sys of G and X
        x      = odeint(GXI_system1, x0, t, args=(p1,p2,p3,Gb,Ib, sol_I))
    elif Input_Index==1:
        x      = odeint(GXI_system2, x0, t, args=(p1,p2,p3,Gb,Ib))
    else:
        print("Please enter a valid index (0: pharmacokinetics model; 1: insulin data)")
        return "Invalid choice error"

    ## Plot the result
    G = x[:,0]
    X = x[:,1]

    if Plot:
        plt.plot(t, G, label='Glucose Model Simulation')
        plt.xlabel("t [min]")
        plt.ylabel("G [mmol/l]")
        plt.grid(True)
        plt.plot(t, Gb*np.ones([len(t)]), '--', label="Basal Glucose Level")
        plt.legend()
        plt.title("Glucose Level G(t) with Minimal Model")
        plt.show()

        #Plot X
        plt.plot(t,X)
        plt.xlabel("t [min]")
        plt.ylabel("X [mU/l]")
        plt.grid(True)
        plt.title("Remote Insulin Level X(t) with Minimal Model")
        plt.show()

    return G, X
