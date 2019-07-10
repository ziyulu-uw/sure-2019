# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: verify the formula for covariance S and cost function V

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from initialization import *
from tensor_tool import transpose
import process_generator
import stat_analysis
from stat_analysis import covariance
import Model_tools
import EigenTest

'''--------------------------Formula of S------------------------------'''
print("********** verify the formula of S is correct ************\n")

## Parameters setting and check they are in the stability region
np.random.seed(122)  # set random seed so the result can be reproduced
G1      = -0.5
G2      = -0.1
G       = np.array([[G1, G2]])
EigenTest.M_eval(G,display=False)

## Find out S from simulation
X_      = process_generator.mass_spring_LQR(X_initial,G,Plot=False) #X_ records x,v at all time from 0 to t1
X       = np.reshape(X_[:,-1,:],[n,2,1])                           #X at the final time
S_sim   = process_generator.S_simulation(G,display=False,Plot=False) 

## Find out S theoretically
S_thm   = Model_tools.S_matrix(G1,G2,returnXiS=False,display=False) 
stat_analysis.Compare_EX2_and_S(X_,S_thm)      #compare and display results

'''--------------------------Formula of S dot----------------------------'''
print("\n*********** verify Vn = tr(S) + r*tr(GSG.T) **************\n")

## Find S dot theoretically
S_dot1  = Model_tools.S_dot(G1,G2,returnS1=True, returnXi=False,display=False) 
S_dot2  = Model_tools.S_dot(G1,G2,returnS1=False,returnXi=False,display=False) 

## Cost from simulation
V       = process_generator.Cost_function(G,X)
stat_analysis.Compare_V_and_Sformula(V,S_thm,G) #compare and display results








