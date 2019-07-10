# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: 1. check the eigenvalues of M are inside unit circle
#              2. 

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
EigenTest.M_eval(G1=1,    G2=2,    display=True)
EigenTest.M_eval(G1=0,    G2=0,    display=True)
EigenTest.M_eval(G1=-0.1, G2=-0.1, display=True)
EigenTest.M_eval(G1=-1,   G2=-2,   display=True)
EigenTest.M_eval(G1=-200, G2=-1,   display=True)
## Plot the stability region of M w.r.t G
EigenTest.Stability_Region()

##--------------------------Formula of S--------------------------------
np.random.seed(1)  # set random seed so the result can be reproduced
X_ = process_generator.mass_spring_LQR(X_initial,G1=-0.1,G2=-0.1,Plot=True)
plt.plot(np.average(X_[:,:,0],axis = 0))

# #### Newton Method with Adaptive Step Size Control
# 
# Evaluate cost function $(V)$: 
# <li> If $V$ increases, change step size $t$ to $\frac{t}{2}$ 
# <li> If $V$ decreases, $t$ is unchanged.

