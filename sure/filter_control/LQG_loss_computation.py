# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program offers some functions that calculate LQG loss and plot

import numpy as np
import matplotlib.pyplot as plt
from LQG_tool import transpose
from loss_computation import compute_loss  # calculate the cost here


def compute_cost_rate(X_val, TrueX_val, G_val, N, r):
    """Xn+1^2 + rUn^2
    @parameters: average X: X is N*2*1 , average U: U is N*1*1
    @return: cost rate"""
    G            = np.reshape(G_val[-1, :], [1, 2])
    X            = np.reshape(np.average(TrueX_val, axis=0), [N + 1, 2, 1])
    X_hat        = np.reshape(np.average(X_val, axis=0), [N + 1, 2, 1])
    U            = G @ X_hat
    U[-1, :, :]  = 0  # we do not take account of U_N into the loss
    d_X          = len(X[0,:,:])
    Y            = np.zeros([len(X),d_X,1])  # Y is used to take values from X1 to X_N-1, X_N = 0
    Y[:-1, :, :] = X[1:, :, :]
    Jn           = np.real((transpose(Y)@Y+r*transpose(U)@U)/(2*len(Y)))
    cost_rate    = np.reshape(Jn, [1, N + 1])[0][:-1]
    return cost_rate


def compute_cost(X_val, TrueX_val, G_val, N, r):
    """call compute_loss in this function of calculate the total cost of LQG simulation
     @parameters: X_val: X estimation value, n*N*2;
                 TrueX_val: real value of X of n paths at N time steps, n*N*2
                 G_val: the value of control G at N time step, N*2"""
    G = np.reshape(G_val, [N+1, 1, 2])
    X = np.reshape(np.average(TrueX_val, axis=0), [N+1, 2, 1])
    X_hat = np.reshape(np.average(X_val, axis=0), [N+1, 2, 1])
    U = G @ X_hat
    U[-1, :, :] = 0  # we do not take account of U_N into the loss
    return compute_loss(X, U, N, r)


def Plot_Cost(X_val_list, TrueX_val_list, G_val_list, t, N, r, num_of_plot, str_list):
    """calculate the cost given the simulation result and G find out using LQG control
    @parameters: X_val_list is a list taking all X_val (X estimation value, n*N*2);
                 TrueX_val_list is a list taking all TrueX_val (real value of X of n paths at N time steps, n*N*2)
                 G_val_list is a list taking all G_val (the value of control G at N time step, N*2)
                 t: time linspace for plotting
                 N: num of time steps
                 r: constant coefficient in loss function
                 X0: initial value of X, 2*1
                 """
    plt.figure(1)
    for i in range(num_of_plot):
        X_val     = X_val_list[i]
        TrueX_val = TrueX_val_list[i]
        G_val     = G_val_list[i]
        cost_rate = compute_cost_rate(X_val, TrueX_val, G_val, N, r)
        # Plot Cost
        plt.plot(t, cost_rate, label = str_list[i])
    plt.xlabel("time")
    plt.ylabel("cost rate")
    plt.title("cost rate w.r.t Time")
    plt.legend()
    plt.show()
    return cost_rate


def after_train_cost(K, G, W, V, A, B, C, x0, v0, n, N):
    """after we have found out K, G in steady state, we want to know the value of loss use this pair of K and G
    @:parameter K 2*1 Kalman filter
                G 1*2 optimal control
                W, V noise"""

    # allocate space for arrays
    Xn_hat = np.zeros([n, 2, 1])          # estimation of X
    Xn = np.zeros([n, 2, 1])              # true values of X
    X_val = np.zeros([n, N + 1, 2])       # a tensor to take all Xn = [xn;vn]
    TrueX_val = np.zeros([n, N + 1, 2])   # true value of X

    # initial state, each matrix has the form: [x0; v0]
    Xn_hat[:, 0, 0] = x0  # put the initial condition into Xn_hat
    Xn_hat[:, 1, 0] = v0
    Xn[:, 0, 0] = x0  # put the initial condition into Xn
    Xn[:, 1, 0] = v0
    X_val[:, 0, 0] = Xn_hat[:, 0, 0].T  # xn
    X_val[:, 0, 1] = Xn_hat[:, 1, 0].T  # vn
    TrueX_val[:, 0, 0] = Xn[:, 0, 0].T  # xn
    TrueX_val[:, 0, 1] = Xn[:, 1, 0].T  # vn

    A = np.real(A)

    # time update
    for i in range(N):
        '''The formulas are from Professor Goodman's notes Kalman Filter Formula'''
        k = i + 1

        # NOISE GENERATION
        # Wn = W[k,:,:,:] # Wn is not used in the simulation
        Vn = V[k, :, :, :]

        # CONTROL GAIN Gn
        Gn = G

        # KALMAN FILTER
        Kn = K

        # update true measurements and estimation
        Z = C @ Xn + Vn
        Zn_hat = C @ (A @ Xn_hat + B @ Gn @ Xn_hat)

        # update true Xn and estimation
        Xn_hat = A @ Xn_hat + B @ Gn @ Xn_hat + Kn @ (Z - Zn_hat)
        Xn = A @ Xn + B @ Gn @ Xn_hat

        # RECORD STATISTICS
        # record Xn_hat = [xn;vn] of n paths
        X_val[:, k, 0] = Xn_hat[:, 0, 0].T  # xn
        X_val[:, k, 1] = Xn_hat[:, 1, 0].T  # vn

        # record Xn = [xn;vn] of n paths
        TrueX_val[:, k, 0] = Xn[:, 0, 0].T  # xn
        TrueX_val[:, k, 1] = Xn[:, 1, 0].T  # vn

    return X_val, TrueX_val



