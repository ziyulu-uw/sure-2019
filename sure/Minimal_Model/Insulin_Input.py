# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program contains parameters and functions to determine I(t)
#
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import numpy as np
#from Insulin_Control import u

class phar_kin:
    """the class about using pharmacokinetics model in paper3 to determine I(t)"""

    t1 = 600
    N = t1 + 1       # every minute is one step
    t = np.linspace(0, t1, N)  # time discretization

    S1_0 = 9294.9
    S2_0 = 9294.9    # (unit: miuU/kg)
    tI   = 50.9847   # (unit: min)
    #tI = 5
    vb   = S1_0/tI
    #vb   = 182.3085  # (unit: miuU/kg/min)
    VI   = 87        # (unit: dl/kg)
    #VI   = 87*100     # unit conversion: ml/kg
    kI   = 0.1905    # (unit: min^-1)
    vB   = 40000     # (unit: miuU/kg/min)
    Ib   = S1_0/(tI * VI * kI)  # basal plasma insulin concentration (unit: mU/l)
    I_0  = 11         #initial value of I

    @staticmethod
    def v(time, u):
        """v(t) is the rate of subcutaneously administered insulin
        :param time
        :type time - float
        :param u control exerted on insulin input"""
        # Notice: here np.abs(t)<1 is the delta(t), approximation of Dirac impulse \
        # I do not know how they determine this delta(t), I tried some parameters \
        # until i found out that when multiplied by 4.8 gives a very similar result to theirs\
        # and if you try 4 or 5, it will give different maximum
        return phar_kin.vb + phar_kin.vB * (np.abs(time)<1) * u

    @staticmethod
    def plot_v():
        """The function is only to show the graph of insulin input v(t)"""
        t = phar_kin.t
        y = phar_kin.vb + phar_kin.vB * (np.abs(t)<1)
        plt.plot(t,y)
        plt.title("Administered Insulin Input v(t) ")
        plt.ylabel("v [$\mu U/kg/min$]")
        plt.xlabel("time min")
        plt.grid(True)
        plt.show()

    @staticmethod
    def pk_model(x, time, u):
        """Insulin pharmacokinetics model is described in paper3 eqn(4-6)
        :param x- [S1, S2, I]^t, 3*1
        :param time
        :type time - float
        :param u control exerted on insulin input"""

        S1 = x[0]
        S2 = x[1]
        I  = x[2]

        temp = (1/phar_kin.tI)
        dS1_dt = -temp * S1 + phar_kin.v(time, u)
        dS2_dt = -temp * S2 + temp * S1
        dI_dt  = -phar_kin.kI * I + temp * (1/phar_kin.VI) * S2
        return [dS1_dt, dS2_dt, dI_dt]

    @staticmethod
    def solveODE(u, Plot):
        """For any given time t, this function of return the value of I(t)
        :param u     - control exerted on insulin input
        :param Plot  - true if you want to see I(t): figure4 in paper3
        :type  Plot  - Boolean
        :return I(t) - sol has form of  [S1, S2, I], we only need to return I here"""

        t = phar_kin.t
        ## Solve the ODE system governing insulin pharmacokinetics
        sol = odeint(phar_kin.pk_model, [phar_kin.S1_0, phar_kin.S2_0, phar_kin.I_0], t, args=(u,))

        ## Plot I(t) if needed
        if Plot:
            plt.plot(t, sol[:,-1], label="Insulin Simulation Result")
            plt.ylabel("insulin I(t) [$\mu$U/ml]")
            plt.xlabel("time [min]")
            plt.title("Insulin Level I(t) with Pharmacokinetics Model")
            plt.plot(t, phar_kin.Ib * np.ones([len(t)]), '--', label="Basal Insulin Level")
            plt.legend()

            #plt.xlim(0,600)
            #plt.ylim(-100,200)
            plt.grid()
            plt.show()

        return sol[:,-1]


    @staticmethod
    def I(time,sol):
        t = phar_kin.t
        ## For any given time, return I at that time
        for i in range(len(t)):
            if t[i]>= time:
                return sol[i]
        return sol[-1]


class insulin_data:
    """a class to use experiment data and linear approximation to determine I(t)"""
    t1 = 200
    N = t1 + 1  # every minute is one step
    t = np.linspace(0, t1, N)  # time discretization

    # Data from experiment
    Insulin_Data = [5, 5, 37, 38, 31, 28, 27, 20, 18.5, 19, 15, 17, 14, 14.5, 17, 14.5, 15, 13.5, \
                    14.5, 13, 8, 6, 5, 5.5, 4, 3.5]
    Time_Data = [0, 5, 6, 9, 11, 12, 14, 17, 19, 22, 24, 27, 29, 33, 36, 39, 41, 46, \
                 56, 67, 75, 95, 105, 145, 175, 200]

    Ib = 5  # basal plasma insulin concentration (unit: mU/l)

    @staticmethod
    def linear_approx_eqn(t, Plot):
        """This function use data points and linear approximation to determine the Insulin Function I(t)
        :param Plot = True if you want to plot I(t)
        :param t is from Parameter.py, time discretization
        :return y - the linear approximation based on data points
        :return ygrad - the derivative of y"""

        y = np.interp(t, insulin_data.Time_Data, insulin_data.Insulin_Data)

        if Plot:
            plt.plot(insulin_data.Time_Data, insulin_data.Insulin_Data, '-o')
            plt.grid(True)
            plt.xlabel("Time [min]")
            plt.ylabel("I [$\mu$U/ml]")
            plt.title("Insulin Level I(t) with Experiment Data")
            plt.show()

        return y, np.gradient(y, t)

    @staticmethod
    def I_linear_data(time):
        """This function to any time point t and return I(t), and I'(t)
        @:param time - any time point between 0 - t1 (end time)
        :type time - float"""
        td = insulin_data.Time_Data
        id = insulin_data.Insulin_Data

        if time == 0:
            return 0

        for i in range(len(td)):
            if td[i] >= time:
                dy = (id[i - 1] - id[i]) / (td[i - 1] - td[i])
                y = dy * (time - td[i]) + id[i]
                return y
        return 0
