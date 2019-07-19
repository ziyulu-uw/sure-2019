# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program computes the loss F and the gradient of F w.r.t. K and G using forward propagation

import numpy as np
from initialization import *
from path_generation import path_generator

def Control_forward(X,X_hat,A,B,C,G,K,r):
    """calculate dF/dG with forward propogation method
    @parameter: X     - simulation result N*2*1
                X_hat - estimation result N*2*1
                A,B,C - constant matrices (seen in initialization.py)
                        A 2*2, B 2*1, C 1*2
                G     - control gain 1*2
                K     - Kalman Filter 2*1
                r     - a constant number in cost rate
    @return:   dF/dG: F is the total cost function, 
                        return the gradient of F w.r.t G 
                        1*2 """
                
    ## create an Identity matrix
    d_X = len(X[0,:,:]) #the dimension of state
    I   = np.eye(d_X)   #an identity matrix
    
    ## initial state
    dXn_dG    = np.zeros([2,2])
    dXnHat_dG = np.zeros([2,2])
    dF_dG     = X[0,:,:].T@dXn_dG     #we will divide dF/dG by N in the end
    
    for i in range(N-1): 
        ## get the estimation at time i from the estimation result
        XnHat     = X_hat[i,:,:]
        Un        = G@XnHat           #find the control and the gradient of U w.r.t G
        dUn_dG    = XnHat.T+G@dXnHat_dG 
        
        ## forward recurrence
        temp      = A@(I-K@C)+B@G     #some matrix algebra that will be used 
        dXnHat_dG = B@X_hat[i,:,:].T+temp@dXnHat_dG+K@C@A@dXn_dG  #d hat{Xn+1}/ dG

        dXn_dG    = A@dXn_dG+B@XnHat.T+B@G@dXnHat_dG              #d Xn+1/dG
        
        ## update the gradient of total cost function
        dF_dG     += X[i+1,:,:].T@dXnHat_dG+r*Un.T@dUn_dG
    return dF_dG/N

def Filter_forward(X,X_hat,Z,A,B,C,G,K,r):
    """calculate dF/dG with forward propogation method
    @parameter: X     - simulation result N*2*1
                X_hat - estimation result N*2*1
                Z     - obeservation CX+V N*1*1
                A,B,C - constant matrices (seen in initialization.py)
                        A 2*2, B 2*1, C 1*2
                G     - control gain 1*2
                K     - Kalman Filter 2*1
                r     - a constant number in cost rate
    @return:   dF/dG: F is the total cost function, 
                      return the gradient of F w.r.t G 
                      1*2 """
                
    ## initial state
    dXn_dK    = np.zeros([2,2])
    dXnHat_dK = np.zeros([2,2])
    dF_dK     = X[0,:,:].T@dXn_dK     #we will divide dF/dG by N in the end
    
    for i in range(N-1): 
        ## get the estimation at time i from the estimation result
        XnHat     = X_hat[i,:,:]
        Un        = G@XnHat           #find the control and the gradient of U w.r.t G
        dUn_dK    = G@dXnHat_dK 
        
        ## forward recurrence
        temp      =  np.reshape(np.diag(Z[i+1,:,:]-C@(A+B@G)@XnHat),[1,1])
        dXnHat_dK = (A+B@G)@dXnHat_dK+temp+K@C@A@(dXn_dK-dXnHat_dK)   #d hat{Xn+1}/dK
        dXn_dK    = A@dXn_dK+B@XnHat.T+B@G@dXnHat_dK                  #d Xn+1/dK
        
        ## update the gradient of total cost function
        dF_dK     += X[i+1,:,:].T@dXnHat_dK+r*Un.T@dUn_dK
    return dF_dK/N

d_X = len(X0)
d_Z = 1
K = np.array([[1.5, 1.9]]).T
G = np.array([[-0.01,-0.01]])

X, Z, U, X_hat = path_generator(X0, A, C, B, G, K, N, W, V, d_X, d_Z, d_U)
dF_dG = Control_forward(X,X_hat,A,B,C,G,K,r)
dF_dK = Filter_forward(X,X_hat,Z,A,B,C,G,K,r)