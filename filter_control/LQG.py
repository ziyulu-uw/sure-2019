# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program is about do LQG control, to compare this gradient descent 
import numpy as np
from numpy.linalg import inv
from LQG_tool import generate_noise, transpose
import numpy.linalg as la

def Sn_backward(A,B,r,n,N):
    '''Sn used in control is calculated backward'''
    I  = np.eye(2)
    Sn = I #S at time step N
    Sn_list = np.zeros([N,2,2])
    r = np.ones(1)*r #need to change r into matrix form
    for i in range(N):
        Sn_list[N-i-1]= Sn              #Sn+1
        Sn = A.T@(Sn-Sn@B@inv(B.T@Sn@B+r)@B.T@Sn)@A+I  #Sn
    return Sn_list

def LQG_simulation(x0,v0,A,B,C,R,S,r,Sn_list,n,N,DoLQG):
    '''the function to do LQG control in simulation
    @parameter: x0, v0 are initial conditions
                A fundamental solution
                B constant matrix exerted on control
                C constant matrix exerted on measurement
                R covariance matrix of process noise
                S covariance matrix of observation noise
                Sn_list a series of matrices that will be used in calculationg control
                        the calculation of Sn's are in function Sn_backward
                n num of paths
                N length of each path
    @return:    X_val n*(N+1)*2 all the x,v
                K_val n*(N+1)*2 all the Kalman Gain
                G_val (N+1)*2 all the Control Gain'''
    
    ## allocate space for arrays
    Zn_hat    = np.zeros([n,2,1])    # predicted observation using old data Z(n+1) = C(A+BG) X_hat(n)
    Xn_hat    = np.zeros([n,2,1])    # estimation of X
    Xn        = np.zeros([n,2,1])    # true values of X
    Pn        = np.zeros([n,2,2])    # Pn: covariance matrix of Xn-^Xn is Tn, Pn = ATnA.T+R
    Kn        = np.zeros([n,2,1])    # Kalman Gain
    Gn        = np.zeros([1,2])      # Control Gain
    Tn        = np.zeros([n,2,2])    # covariance matrix of prediction error

    K_val     = np.zeros([n,N+1,2])    # a tensor to take all K11 and K12 values
    G_val     = np.zeros([N+1,2])      # a tensor to take all G val at all time
    X_val     = np.zeros([n,N+1,2])    # a tensor to take all Xn = [xn;vn]
    TrueX_val = np.zeros([n,N+1,2])    # true value of X

    ## create n identity matrix I=[[1,0];[0,1]]
    I         = np.zeros([n,2,2])
    I[:,0,0]  = 1
    I[:,1,1]  = 1
    
    ## initial state, each matrix has the form: [x0; v0]
    Xn_hat[:,0,0] = x0                   # put the initial condition into Xn_hat
    Xn_hat[:,1,0] = v0 
     
    Xn[:,0,0] = x0                       # put the initial condition into Xn
    Xn[:,1,0] = v0 
    
    X_val[:,0,0]= Xn_hat[:,0,0].T              #xn
    X_val[:,0,1]= Xn_hat[:,1,0].T              #vn
        
    TrueX_val[:,0,0]= Xn[:,0,0].T              #xn
    TrueX_val[:,0,1]= Xn[:,1,0].T              #vn
    
    ## take the tranpose of some constant matrices that will be used later    
    C_T = C.T
    A = np.real(A)
    A_T = A.T
    B_T = B.T
    
    ## generate n paths of noises, each of length N
    # W = generate_noise(R,n,N+1) #process noise: R-cov  [Wn is not used in the simulation]
    V = generate_noise(S,n,N+1) #observation nois: S-cov
    
    r = np.ones([1,1])*r     #need to reshape r to 3d matrix form
    ## time update
    for i in range(N):
        '''The formulas are from Professor Goodman's notes Kalman Filter Formula'''
        k = i+1 
        
        ##NOISE GENERATION
        # Wn = W[k,:,:,:] # Wn is not used in the simulation
        Vn = V[k,:,:,:]
        
        ##Update CONTROL GAIN Gn
        Sn = Sn_list[N-k,:,:]
        if DoLQG:
            Gn          = -inv(B_T@Sn@B+r)@B_T@Sn@A
        
        ## KALMAN FILTER [Approach.1]
        #update Pn: covariance matrix of Xn-^Xn is Tn, Pn = ATnA.T+R
        #Pn          = A@(Pn-Kn@C@Pn)@A_T+R  #Pn+1
        #update Kalman Gain Kn
        #Kn          = Pn@C_T@inv(C@Pn@C_T+S) #S is the covariance matrix of obseration noise
        
        ## KALMAN FILRWE UPDATE[Approach.2]
        # ----DEBUGGING: Tn shoud be equal to ATA+R-------------
        #update covariance matrix of Yn
        Tn          = np.real((A-Kn@C@A)@Tn@transpose(A-Kn@C@A)+(I-Kn@C)@R@transpose(I-Kn@C)+Kn@S@transpose(Kn)) 
        #print("ATA+R:",(A@Tn@A.T+R)[0,:,:])
        #print("Pn:", Pn[0,:,:])
        Kn          = (A@Tn@A_T+R)@C_T@la.inv(C@A@Tn@A_T@C_T+C@R@C_T+S)
        #print("CATA",(C@A@Tn@A_T@C_T+C@R@C_T+S)[0,:,:])
        #print("CPC",(C@Pn@C_T+S)[0,:,:])
        
        #update true measurements and estimation
        Z           = C@Xn + Vn
        Zn_hat      = C@(A@Xn_hat+B@Gn@Xn_hat)
        
        #update true Xn and estimation
        Xn_hat      = A@Xn_hat+B@Gn@Xn_hat + Kn@(Z-Zn_hat)
        Xn          = A@Xn + B@Gn@Xn_hat 
                
        ## RECORD STATISTICS
        #record the estimation of n paths
        #pos[:,k,0]  = Xn_hat[:,0,0].T             #pos[i,:,:] is one trajectory, 
                                                   #pos[:,k,:] is all pos of n paths at time k
        #record the values of Kn of n paths
        K_val[:,k,0]= Kn[:,0,0].T                  #K11
        K_val[:,k,1]= Kn[:,1,0].T                  #K12
        
        #record the values of Kn of n paths
        G_val[k,0]= Gn[0,0]                 #G11
        G_val[k,1]= Gn[0,1]                 #G12
        
        #record Xn_hat = [xn;vn] of n paths
        X_val[:,k,0]= Xn_hat[:,0,0].T              #xn
        X_val[:,k,1]= Xn_hat[:,1,0].T              #vn
        
        #record Xn = [xn;vn] of n paths
        TrueX_val[:,k,0]= Xn[:,0,0].T              #xn
        TrueX_val[:,k,1]= Xn[:,1,0].T              #vn
        
    return K_val,G_val,X_val,TrueX_val

