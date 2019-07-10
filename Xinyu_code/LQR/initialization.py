# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: initialize all the parameters

import numpy as np
from numpy import linalg as la

## step size and paths number
h          = 1e-2                                 #time step size in the process \
                                                  #used to decide the covariance matrix R of Wn
t1         = 2                                   #end time
N          = int(t1/h+1)                          #the num of steps of each path
t          = np.linspace(0, t1, N)                #time steps (0, t1, t1/h+1)
n          = int(1e+4)                            #num of paths

## coefficient in the noise
sigma      = 0.1                                  #coefficient in the noise

## constant parameters in alpha
k          = 0.1
m          = 1
gamma      = 0.05
alpha      = np.zeros([2,2])
alpha[0,1] = 1
alpha[1,0] = -k/m
alpha[1,1] = -gamma/m

## The derivative of xi_G
xi_G1    = np.array([[1,0,0],[0,1,0]])
xi_G2    = np.array([[0,1,0],[0,0,1]])
    
## find xi_R (mathematical proof in Mass_Spring_Model_v3)
w, U       = la.eig(alpha)                        #spectral decomposition: dot(a[:,:], U[:,i]) = w[i] * U[:,i] 
                                                  #w are eigenvalues, the columns of U are eigenvectors
Diag       = np.diag(np.exp(w*h))                 #take the exponential of eigenvalue*dt and put them on the diagonal
A          = np.real(U@Diag@la.inv(U)) 
a11        = A[0,0]
a12        = A[0,1]
a21        = A[1,0]
a22        = A[1,1]
w1         = w[0]                                 #eigenvalues
w2         = w[1]
e1         = np.exp(2*w1*h) - 1                   #temp1
e2         = np.exp(2*w2*h) - 1                   #temp2
e3         = np.exp((w1+w2)*h) - 1                #temp3
R          = np.zeros([2,2])
R[0,0]     = np.real(e1/(2*w1)+e2/(2*w2)-2*e3/(w1+w2))     #assign values
R[1,0]     = np.real(e1/2+e2/2-e3)
R[0,1]     = R[1,0]
R[1,1]     = np.real(w1*e1/2+w2*e2/2-2*w1*w2*e3/(w1+w2))
R          = np.real(R*(sigma/(w2-w1))**2)
xi_R       = np.zeros([3,1])
xi_R[0,0]  = R[0,0]
xi_R[1,0]  = R[1,0]
xi_R[2,0]  = R[1,1]

##Constant matrix exerted Control
B          = np.zeros([2,1])                      
B[1][0]    = 1

## parameter in cost function
r          = 500  

## initial condition
x0         = 1
v0         = 0
X_initial  = np.zeros([n,2,1])                    
X_initial[:,0,0]    = x0
X_initial[:,1,0]    = v0