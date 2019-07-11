# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: analysis statistics such as average, covariance matrix

import numpy as np
from tensor_tool import transpose
from initialization import *
import Model_tools

def covariance(X):
    '''calculate the covariance matrix
    @parameter: given X at time t (n*2*1), return the covariance matrix'''
    
    ## calculate the corresponding corvariance 
    EX    = np.average(X,axis = 0)                 #get the expectation of X
    temp  = (X-EX)@(transpose(X)-EX.T)             #get the matrices
    S     = np.real(np.average(temp,axis=0))       #take the average and get the covariance matrix
    #print()
    return S

def Compare_EX2_and_S(X_,S):
    '''Compare E|X|^2 and trace of S
    @parameter: X_ is n*N*2 tensor taken all the values in the process
    we only need the values at end time
    S is the covariance matrix'''
    X       = np.reshape(X_[:,-1,:],[n,2,1])      #take the data we need
    X_T     = transpose(X)                        #transform to the correct shape
    EX2     = np.real(np.average(X_T@X,axis=0))
    print("E|X|^2 (simulation):              %.10f"%EX2[0,0])
    print("Trace of S (theoretical):         %.10f"%np.trace(S))
 
def Compare_V_and_Sformula(V,S,G):
    '''Compare the derivative cost function and the theoretical formula evolving S
    @parameter: V is the cost from simulation at time t1
    S is the covariance matrix, S_dot is the derivative of S w.r.t G'''
    print("Cost (simulation):                %.10f" % V)
    print("tr(S+rGSG.T) (theoretical):       %.10f" % (np.trace(S)+r*np.trace(G@S@G.T)))
    print("r = %.2f"%r)

def Compare_dS_formula_and_FD(G):
    '''Compaore dS formula and the approx using finite difference'''
    
    G1 = G[0,0]
    G2 = G[0,1]
    S  = Model_tools.S_matrix(G1,G2,returnXiS=False,display=False)

    ## Differentiate w.r.t to G1
    # Finite Difference Approximation
    h  = 1e-10     #a small interval used in finite difference approximation
    G1_f = h+G1
    S_f  = Model_tools.S_matrix(G1_f,G2,returnXiS=False,display=False)
    S_D  = (S_f-S)/h
    print("dS w.r.t G1\n Finite Difference Approx:\n",S_D)
    dS   = Model_tools.S_dot(G1,G2,returnS1=True,returnXi=False,display=False)
    print(" dS = inv(I-D) xi_u:\n",dS)
    #S_dot2(G1,G2,returnS1=True,returnXi=False,display=True)
    print()
    ## Differentiate w.r.t to G1
    G2_f = h+G2
    S_f  = Model_tools.S_matrix(G1,G2_f,returnXiS=False,display=False)
    S_D  = (S_f-S)/h 
    print("dS w.r.t G2\n Finite Difference Approx:\n",S_D)
    # dS formula
    dS   = Model_tools.S_dot(G1,G2,returnS1=False,returnXi=False,display=False)
    print(" dS = inv(I-D) xi_u:\n",dS)
    #S_dot2(G1,G2,returnS1=True,returnXi=False,display=True)
    print()
    
   