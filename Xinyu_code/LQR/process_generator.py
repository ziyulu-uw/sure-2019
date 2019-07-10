# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: generate the process

from initialization import *
import numpy as np
import matplotlib.pyplot as plt
from stat_analysis import covariance
from tensor_tool import transpose

def mass_spring_LQR(X_initial,G,Plot):
    '''Mass spring process with LQR control'''
    ##Allocate space
    X_mat = np.zeros([n,N,2])                    #a matrix to record all the trajectory of X=[xn,vn]
    #X_mat[n,:,:] is a complete path, at each time has the form [x0, v0]
    X       = X_initial.copy()                   #initial condition
    
    for i in range(N): #time evolution
        ##generate noise W
        W = np.random.multivariate_normal(np.array([0,0]),np.real(R),n) #mean is [0,0] for Wn
        W = np.reshape(W,[n,2,1])                                       #reshape W to make it n*2*1
        U = G@X
        #print(U)
        ##time evolution with noise
        X = A@X+W+B@U
        X_mat[:,i,0] = X[:,0,0].T
        X_mat[:,i,1] = X[:,1,0].T
    #print(np.size(X_mat))
    
    ##Plot the process (average)
    if Plot:    
        plt.plot(t,np.average(X_mat[:,:,0],axis = 0),label="$x$")
        plt.title("The average position $x$ of %s paths with $G_1$ =%.2f and $G_2$ =%.2f"%(n,G[0,0],G[0,1]))
        plt.legend()
        plt.show()
    return np.real(X_mat) #the imaginary part of X are almost zero (in theory should be zero)

def S_simulation(G,display,Plot):
    '''generate a process then calculation the covariance based on simulation
    @return: the covariance matrix S from simulation'''
    X_      = mass_spring_LQR(X_initial,G,Plot)
    X_final = np.reshape(X_[:,-1,:],[n,2,1]) #reshape it to n*2*1
    S1      = covariance(X_final) # get the covariance of X at end time t1
    if display:
        print("Covariance matrix S from simulation:\n",S1)
        print("(The number of paths:%d)\n(end time: %d)\n"%(n,t1))
    return S1

def Cost_function(G,X):
    '''cost function Vn = Xn^2+rUn^2'''
    U = G@X
    U_T = transpose(U)
    X_T = transpose(X)
    #print("U^2",r*np.average(U_T@U,axis=0))
    V = np.average(X_T@X+r*U_T@U,axis=0)[0,0]
    return np.real(V)
    
    
    
    
    
    