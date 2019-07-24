# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program is about do LQG control and Kalman Filter Estimation, to compare this gradient descent 

import numpy as np
from initialization import *
                                 #params are set here: Constant matrices: A,B,C;
                                 # Cov: R,S; N-num of time step
import LQG                       #simulation with LQG is done here
from LQG_tool import Plot_K,Plot_G,Plot_X,Plot_Cost,display     #This program offers some functions that used in LQG and plot
n = 1000  # number of paths

DoLQG = True #If you only want to see Kalman Filter Estimation, set here to be False;
              # If you want to do LQG, set here to be True
              
## Find out Sn (a series of matrices that will be used in Gn calculation)
#  Sn is determined by a matrix Riccati difference equation backward in time
Sn_list = LQG.Sn_backward(A,B,r,n,N)

## Simulate the process with LQG control
K_val,G_val,X_val,TrueX_val = LQG.LQG_simulation(x0,v0,A,B,C,R,S,r,Sn_list,n,N,DoLQG)

## Plot K,G,and X
Plot_K(K_val,t)
Plot_X(X_val,t)
if DoLQG:
    Plot_G(G_val,t)
    cost = Plot_Cost(X_val,TrueX_val,G_val,t,r,N,X0)
display(G_val,K_val) #display the G and K in the steady state
