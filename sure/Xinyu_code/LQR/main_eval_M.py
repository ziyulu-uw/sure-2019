# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: check the eigenvalues of M are inside unit circle

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from initialization import *
import tensor_tool 
from tensor_tool import transpose
import EigenTest
import process_generator

##-----------------------Eigenvalues of M------------------------------

## Display some eigenvalues of M given G1 and G2
EigenTest.M_eval(G = np.array([[1,2]]),      display=True)
EigenTest.M_eval(G = np.array([[0,0]]),      display=True)
EigenTest.M_eval(G = np.array([[-0.1,-0.1]]),display=True)
EigenTest.M_eval(G = np.array([[-1,-2]]),    display=True)
EigenTest.M_eval(G = np.array([[-200,-1]]),  display=True)

## Plot the stability region of M w.r.t G
EigenTest.Stability_Region()

##--------------------------Formula of S--------------------------------
np.random.seed(1)  # set random seed so the result can be reproduced
G  = np.array([[-0.5,-0.1]])
X_ = process_generator.mass_spring_LQR(X_initial,G,Plot=True)

