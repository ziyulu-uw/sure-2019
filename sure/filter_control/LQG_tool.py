# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program offers some functions that used in LQG and plot

import numpy as np
import matplotlib.pyplot as plt
from loss_computation import compute_loss #calculate the cost here

def transpose(x):
    '''a function used to take transpose of each matrix inside a tensor x
     x is a tensor n*d*c, return n*c*d'''
    d = len(x[0,:,0])                       #num of rows of each matrix
    c = len(x[0,0,:])                       #num of columns of each matrix
    n = len(x[:,0,0])                       #num of matrices in the tensor x
    X_T = np.zeros([n,c,d],dtype = complex)  #allocate space
    for i in range(d):
        X_T[:,:,i]=x[:,i,:]
    return X_T

def generate_noise(R,n,N):
    '''generate process noise or observation noise
    @parameter: covariance matrix d*d R
                the number of paths n
    @return: n*d*1 gaussian noise with mean 0, covariance R'''
    d = len(R)
    mean = np.zeros([1,d])[0] #mean has to be 1d
    W = np.random.multivariate_normal(mean,np.real(R),[n,N])
    W = np.reshape(W,[N,n,d,1])
    return W

def Plot_G(G_val,t):
    ## Plot Gn
    plt.plot(t,G_val[:-1,0],label = "$G_1$")
    plt.plot(t,G_val[:-1,1],label = "$G_2$")
    plt.xlabel("time")
    plt.ylabel("G")
    plt.title("Control Gain w.r.t Time")
    plt.legend()
    plt.show()

def Plot_K(K_val,t):
    ## Plot Kn
    K_avg = np.average(K_val,axis = 0)
    plt.plot(t,K_avg[:-1,0],label = "$K_1$")
    plt.plot(t,K_avg[:-1,1],label = "$K_2$")
    plt.xlabel("time")
    plt.ylabel("K")
    plt.title("Kalman Gain w.r.t Time")
    plt.legend()
    plt.show()

def Plot_X(X_val,t):
    ## Plot Xn
    X_avg = np.average(X_val,axis = 0)
    plt.plot(t,X_avg[:-1,0],label = "$X_1$")
    plt.plot(t,X_avg[:-1,1],label = "$X_2$")
    plt.xlabel("time")
    plt.ylabel("X")
    plt.title("X w.r.t Time")
    plt.legend()
    plt.show()
    
def compute_cost_rate(X,U):
    '''Xn+1^2 + Un^2
    @parameters: average X: X is N*2*1 , average U: U is N*1*1
    @return: cost rate'''
    d_X = len(X[0,:,:])
    Y = np.zeros([len(X),d_X,1]) #Y is used to take values from X1 to X_N-1, X_N = 0
    Y[:-1,:,:]=X[1:,:,:]
    return (transpose(Y)@Y+transpose(U)@U)/(2*len(Y))

def Plot_Cost(X_val,TrueX_val,G_val,t,r,N,X0):
    '''calculate the cost given the simulation result and G find out using LQG control'''
    G = np.reshape(G_val[-1,:],[1,2])
    X = np.reshape(np.average(TrueX_val,axis = 0),[N+1,2,1])
    X_hat= np.reshape(np.average(X_val,axis = 0),[N+1,2,1])
    U = G@X_hat
    U[-1,:,:] = 0 #we do not take account of U_N into the loss
    Jn = np.real(compute_cost_rate(X,U))
    cost = compute_loss(X, U, N, r)
    ## Plot Cost
    plt.plot(t,np.reshape(Jn,[1,N+1])[0][:-1])
    plt.xlabel("time")
    plt.ylabel("cost rate")
    plt.title("cost rate w.r.t Time")
    plt.show()
    ## Display Total Cost
    print("Total Cost:",cost)
    return cost

def display(G_val,K_val):
    print("G1:",'{:.2e}'.format(G_val[-1,0]),"\t","G2:",'{:.2e}'.format(G_val[-1,1]))
    K = np.average(K_val,axis = 0)
    print("K1:",'{:.2e}'.format(K[-1,0]),"\t","K2:",'{:.2e}'.format(K[-1,1]))
