# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: check the eigenvalues of M are inside unit circle

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from initialization import *

def M_eval(G,display):
    '''a function to show the eigenvalues of M given parameters G1 and G2
    @param: G1 and G2 are the paramters we want to find out in order to have F(G) = 0
    display is either True or False, about whether to display the eigenvalues of M
    @return: either True or False
            True-> the pair G1 and G2 make M stable
            False-> otherwise'''
    
    ## Check whether the eigenvalues of M are inside the unit circle
    M          = A+B@G
    e_val,e_vec= la.eig(M)                      #spectral decomposition: dot(M[:,:], e_vec[:,i]) = e_val[i] * e_vec[:,i] 
                                                #e_val are eigenvalues, the columns of e_vec are eigenvectors
    
    if (np.array([la.norm(e_val[0]),la.norm(e_val[1])]) < np.array([1,1])).all():
        Plot = True
    else:
        Plot = False
        
    if display:
        print("parameters G1 and G2: %15.2f, %15.2f"%(G1,G2))
        print("eigenvalues of M    : %5.5f+%5.5fj, %5.5f+%5.5fj"%(e_val[0].real,e_val[0].imag,e_val[1].real,e_val[1].imag))
        if Plot == True:
            print("STABLE")
        else:
            print("UNSTABLE")
        print("----------------------------------------------------------")
        print()
    return Plot


def Stability_Region():
    ''' Plot a scatter graph to see the stability region of M'''
    
    G1_list = np.linspace(-500,100,150)
    G2_list = np.linspace(-5,2,150)
    posX    = [] #take all the value of G1 in the pair that makes M stable
    posY    = [] #take all the value of G2 in the pair that makes M stable
    
    for G1 in G1_list:
        for G2 in G2_list:
            Plot = M_eval(G1,G2,display=False)
            if Plot:
                posX.append(G1)
                posY.append(G2)
    
    # Plot the stability region of M w.r.t G
    plt.scatter(posX,posY,s=5)
    plt.xlim([-500,100])
    plt.ylim([-5,2])
    plt.plot(G1_list*0,G2_list,'r--',label = "x=0")
    plt.plot(G1_list,G1_list*0.01,'b--',label = "y=0.01x")
    plt.plot(G1_list,G1_list*0.005-2,'g--',label = "y=0.005x-2")
    plt.xlabel("$G_1$")
    plt.ylabel("$G_2$")
    plt.title("The stability region of $M$ w.r.t paramter $G$")
    plt.legend()
    plt.show()


