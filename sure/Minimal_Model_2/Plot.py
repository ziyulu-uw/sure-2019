# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: This module is used for plot some functions

import matplotlib.pyplot as plt

def Plot_Ra(t_list, Ra_list):
    """a function used to plot glucose rate Ra
    @:param t_list:   time discretization
    @:param Ra_list:  glucose disappearance rate """

    plt.plot(t_list, Ra_list)
    plt.title("Glucose Rate of Appearance $R_a$")
    plt.xlabel("min")
    plt.ylabel("$R_a$ (mg/kg/min)")
    plt.show()
    return Ra_list


def Plot_I(t_list, I_list):
    """a function used to plot glucose rate Ra
    @:param t_list:   time discretization
    @:param I_list:   insulin function """

    plt.plot(t_list,I_list)
    plt.title("Insulin $I(t)$")
    plt.xlabel("min")
    plt.ylabel("$I$ (mU/l)")
    plt.show()


def Plot_G(t_list, G):
    """a function used to plot glucose rate Ra
    @:param t_list:   time discretization
    @:param I_list:   insulin function """

    plt.plot(t_list,G,'-')
    plt.title("Glucose $G(t)$")
    plt.xlabel("min")
    plt.ylabel("$G$ (mg/dl)")
    plt.show()


def Plot_X(t_list, sol):
    plt.plot(t_list,sol[0],'-')
    plt.title("$X(t)$")
    plt.xlabel("min")
    plt.ylabel("$X$ (mg/dl)")
    plt.show()

