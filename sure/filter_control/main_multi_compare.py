# Author: Xinyu Li, Ziyu Lu
# Email: xl1796@nyu.edu, zl1546@nyu.edu
# Date: July 2019
# Description: This program is to compare the result given by different initialization parameters
import numpy as np
import LQG                   # simulation with LQG is done here
from init_func import init   # initial parameter are set using the func in this module
from LQG_tool import generate_noise, Plot_X  # noise generation
from LQG_loss_computation import compute_cost,after_train_cost
import matplotlib.pyplot as plt
from simplified_wrappers import wrapper
import performance_test


def dt_test():
    """This function is used to compare the cost, filter, optimal control given by different dt"""
    ## Find out the relation w.r.t dt
    k         = 1  # spring constant
    gamma     = 0.1  # friction coefficient
    sigma     = 0.1  # noise coefficient in SDE
    x0        = 1    # initial condition
    v0        = 0
    r         = 1    # scaling factor in the cost
    obv_noise = 0.3  #  covariance of observation noise
    dt_list   = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0] # Please choose what can divide t1
    n         = 1000  # number of paths

    ## allocate space
    #  For LQG
    K1_list = []
    K2_list = []
    G1_list = []
    G2_list = []
    cost_list = []
    test_cost = []
    #  For SGD
    K1_l   = []
    K2_l   = []
    G1_l   = []
    G2_l   = []
    cost_l = []

    for dt in dt_list:
        ## Initialize the params
        x0, v0, t, X0, A, B, C, S, R, d_X, d_U, d_Z, r, N = init(k, gamma, dt, sigma, x0, v0, r, obv_noise)

        ###-------------------------------- Find Filter and Control using LQG-------------------------------------------
        ## Find out Sn (a series of matrices that will be used in Gn calculation)
        Sn_list = LQG.Sn_backward(A, B, r, n, N)

        ## Generate process and observation noise
        W       = generate_noise(R, n, N+1)  # process noise: R-cov  [Wn is not used in the simulation]
        V       = generate_noise(S, n, N+1)  # observation noise: S-cov

        ## Simulate the process with LQG control
        K_val, G_val, X_val, TrueX_val = LQG.LQG_simulation(x0, v0, A, B, C, R, S, r, Sn_list, W, V, n, N, DoLQG=True)
        cost    = compute_cost(X_val, TrueX_val, G_val, N, r)
        K       = np.reshape(np.average(K_val, axis=0)[-1, :], [2, 1])
        G       = np.reshape(G_val[-1, :], [1, 2])

        ## Calculate the after train cost
        G_val_ = np.zeros([N + 1, 2])
        G_val_[:, 0] = G[0, 0]          # in this case, G_val_ from t=0 to t=N should have same G1 and G2 values
        G_val_[:, 1] = G[0, 1]
        X_val_, TrueX_val_ = after_train_cost(K, G, W, V, A, B, C, x0, v0, n, N)
        cost2 = compute_cost(X_val_, TrueX_val_, G_val_, N, r)

        ###-------------------------------- Find Filter and Control using SGD-------------------------------------------
        K_SGD, G_SGD, F_SGD = wrapper(X0, A, C, B, G, K, N, S, R, d_X, d_Z, d_U, r, n=600, act=True, L=[100, 200, 400], g=0.1, s_l=[1, 4, 10, 20, 27], alpha=0.1, momentum=0, M=1, which='Adam')
        # K_SGD 2*1, G_SGD 1*2, F_SGD float

        ## find the testing result of SGD
        test_loss = performance_test.test(X0, A, C, B, G_SGD, K_SGD, N, R, S, r, d_X, d_Z, d_U, n=1000)

        ## Record the value
        cost_list.append(cost)
        test_cost.append(cost2)
        K1_list.append(K[0, 0])
        K2_list.append(K[1, 0])
        G1_list.append(G[0, 0])
        G2_list.append(G[0, 1])
        K1_l.append(K_SGD[0, 0])
        K2_l.append(K_SGD[1, 0])
        G1_l.append(G_SGD[0, 0])
        G2_l.append(G_SGD[0, 1])
        cost_l.append(test_loss)

        Plot_X(X_val, t)
    return dt_list, K1_list, K2_list, K1_l, K2_l, G1_list, G1_l, G2_list, G2_l, cost_list, cost_l, test_cost


def Plot_dt_test(dt_list, K1_list, K2_list, K1_l, K2_l, G1_list, G1_l, G2_list, G2_l, cost_list, cost_l, test_cost):
    """a function to plot and display values"""

    print("-----------------------LQG Control with Different dt---------------------------")
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} " \
          .format("dt", dt_list[0], dt_list[1], dt_list[2], dt_list[3], dt_list[4], dt_list[5]))
    print("{:12s}   {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} "\
          .format("LQG K1", K1_list[0],  K1_list[1],  K1_list[2],  K1_list[3],  K1_list[4], K1_list[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} " \
          .format("SGD K1", K1_l[0], K1_l[1], K1_l[2], K1_l[3], K1_l[4], K1_l[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} " \
          .format("LQG K2", K2_list[0], K2_list[1], K2_list[2], K2_list[3], K2_list[4], K2_list[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} " \
          .format("SGD K2", K2_l[0], K2_l[1], K2_l[2], K2_l[3], K2_l[4], K2_l[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} " \
          .format("LQG G1", G1_list[0], G1_list[1], G1_list[2], G1_list[3], G1_list[4], G1_list[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} " \
          .format("SGD G1", G1_l[0], G1_l[1], G1_l[2], G1_l[3], G1_l[4], G1_l[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} " \
          .format("LQG G2", G2_list[0], G2_list[1], G2_list[2], G2_list[3], G2_list[4], G2_list[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} " \
          .format("SGD G2", G2_l[0], G2_l[1], G2_l[2], G2_l[3], G2_l[4], G2_l[5]))
    #print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} " \
    #      .format("LQG cost", cost_list[0], cost_list[1], cost_list[2], cost_list[3], cost_list[4], cost_list[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} " \
          .format("LQG cost", test_cost[0], test_cost[1], test_cost[2], test_cost[3], test_cost[4], test_cost[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e} " \
          .format("SGD cost", cost_l[0], cost_l[1], cost_l[2], cost_l[3], cost_l[4], cost_l[5]))


    plt.figure(1)
    plt.plot(dt_list, K1_list,'-+', label="K1")
    plt.plot(dt_list, K2_list,'-+', label="K2")
    plt.xlabel("dt")
    plt.ylabel("K in steady state")
    plt.title("Kalman Filter K with respect to time discretization dt")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(dt_list, G1_list,'-+', label="G1")
    plt.plot(dt_list, G2_list,'-+', label="G2")
    plt.xlabel("dt")
    plt.ylabel("G in steady state")
    plt.title("Control Gain G with respect to time discretization dt")
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(dt_list, cost_list, '-+', label="transitional Kn, Gn")
    plt.plot(dt_list, test_cost, '-+', label="fixed K, G from steady state")
    plt.xlabel("dt")
    plt.ylabel("cost rate")
    plt.title("Cost rate with respect to time discretization dt")
    plt.legend()
    plt.show()


dt_list, K1_list, K2_list, K1_l, K2_l, G1_list, G1_l, G2_list, G2_l, cost_list, cost_l, test_cost = dt_test()
Plot_dt_test(dt_list, K1_list, K2_list, K1_l, K2_l, G1_list, G1_l, G2_list, G2_l, cost_list, cost_l, test_cost)
