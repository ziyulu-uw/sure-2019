# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: analysis statistics such as average, covariance matrix

import numpy as np
from initialization import *
from Model_tools import S_matrix
from Model_tools import S_dot
from Model_tools import H_matrix

def cost_fun(G):
    G1 = G[0]
    G2 = G[1]
    S = S_matrix(G1,G2,returnXiS=False,display=False)
    
    G = np.reshape(G,[1,2])
    return np.trace(S)+np.trace(r*G@S@G.T)

def cost_der(G):
    '''@parameter: G = np.array([G1,G2]) 
    Caution: G is not a matrix here'''
    G1 = G[0]
    G2 = G[1]
    G  = np.array([[G1,G2]]) 
    xiS= S_matrix(G1,G2,returnXiS=True,display=False)
    xiS_Dot1 = S_dot(G1,G2,returnS1=True,returnXi=True,display=False)    
    xiS_Dot2 = S_dot(G1,G2,returnS1=False,returnXi=True,display=False)
    H      = H_matrix(G1,G2)
    Grad1  = q@xiS_Dot1+r*(H@xiS_Dot1+2*G@xi_G1@xiS)
    Grad2  = q@xiS_Dot2+r*(H@xiS_Dot2+2*G@xi_G2@xiS)
    return np.array([Grad1[0,0],Grad2[0,0]])

#l=cost_der(np.array([-0.01,-0.01]))