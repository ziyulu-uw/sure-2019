# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program offers some functions that used in LQG and plot

import numpy as np
import matplotlib.pyplot as plt

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
    @return: N*n*d*1 gaussian noise with mean 0, covariance R'''
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
    plt.grid()
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
    plt.grid()
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
    plt.grid()
    plt.show()


def display(G_val,K_val):
    print("G1:",'{:.2e}'.format(G_val[-1,0]),"\t","G2:",'{:.2e}'.format(G_val[-1,1]))
    K = np.average(K_val,axis = 0)
    print("K1:",'{:.2e}'.format(K[-1,0]),"\t","K2:",'{:.2e}'.format(K[-1,1]))

def Plot_SampleX(t, TrueX_val_, X_val, V):
    import matplotlib.pyplot as plt
    plt.plot(t, X_val[10, :-1, 0], '--', linewidth=1, label="sample trajectory 1")
    plt.plot(t, X_val[1, :-1, 0], '--', linewidth=1, label="sample trajectory 2")
    plt.plot(t, X_val[5, :-1, 0], '--', linewidth=1, label="sample trajectory 3")
    plt.plot(t, np.average(X_val, axis=0)[:-1, 0], label="Expected Value of Mass Position")
    plt.xlabel("Time")
    plt.ylabel("Mass Position x")
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(t, X_val[10, :-1, 1], '--', linewidth=1, label="sample trajectory 1")
    plt.plot(t, X_val[1, :-1, 1], '--', linewidth=1, label="sample trajectory 2")
    plt.plot(t, X_val[5, :-1, 1], '--', linewidth=1, label="sample trajectory 3")
    plt.plot(t, np.average(X_val, axis=0)[:-1, 1], label="Expected Value of Mass Velocity")
    plt.xlabel("Time")
    plt.ylabel("Mass Velocity v")
    plt.grid()
    plt.legend()
    plt.show()

    idx = 12
    # meas1 = TrueX_val_[idx,1:,0]+ np.reshape(V[:-1,idx,:,:],  len(V[:-1,idx,:,:]))
    # plt.plot(t,meas1,'.',label = "observation")
    # plt.plot(t,TrueX_val_[idx,:-1,0],'--',linewidth = 1, label="True Trajectory")
    # plt.plot(t, X_val[idx,:-1,0],linewidth = 1,label="LQG estimation")
    idx = 13
    # meas1 = TrueX_val_[idx,1:,0]+ np.reshape(V[:-1,idx,:,:],  len(V[:-1,idx,:,:]))
    # plt.plot(t,meas1,'.',label = "observation")
    # plt.plot(t,TrueX_val_[idx,:-1,0],'--',linewidth = 1, label="True Trajectory")
    # plt.plot(t, X_val[idx,:-1,0],linewidth = 1,label="LQG estimation")
    idx = 140
    meas1 = TrueX_val_[idx, 1:, 0] + np.reshape(V[:-1, idx, :, :], len(V[:-1, idx, :, :]))
    plt.plot(t, TrueX_val_[idx, :-1, 0], '--', linewidth=1, label="True Trajectory")
    plt.plot(t, X_val[idx, :-1, 0], linewidth=1, label="LQG estimation")
    plt.plot(t[1:], meas1[:-1], '.', label="noisy observation")
    plt.xlabel("Time")
    plt.ylabel("Mass Position x")
    plt.grid()
    plt.legend()
    plt.show()