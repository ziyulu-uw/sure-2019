# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program is to compare the result given by different initialization parameters
import numpy as np
import LQG                   # simulation with LQG is done here
from init_func import init   # initial parameter are set using the func in this module
from LQG_tool import generate_noise  # noise generation
from LQG_loss_computation import compute_cost,after_train_cost
import matplotlib.pyplot as plt

def dt_test():
    """This function is used to compare the cost, filter, optimal control given by different dt"""
    ## Find out the relation w.r.t dt
    k         = 0.5
    gamma     = 0.1
    sigma     = 0.1
    x0        = 1
    v0        = 0
    r         = 1.0
    obv_noise = 0.3
    dt_list   = [0.05, 0.1, 0.25, 0.5, 0.8, 1.0] # Please choose what can divide t1
    n         = 1000  # number of paths

    ## allocate space
    K1_list = []
    K2_list = []
    G1_list = []
    G2_list = []
    cost_list = []
    test_cost = []

    for dt in dt_list:
        ## Initialize the params
        x0, v0, t, X0, A, B, C, S, R, d_X, d_U, d_Z, r, N = init(k, gamma, dt, sigma, x0, v0, r, obv_noise)

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

        ## Record the value
        cost_list.append(cost)
        test_cost.append(cost2)
        K1_list.append(K[0,0])
        K2_list.append(K[1,0])
        G1_list.append(G[0, 0])
        G2_list.append(G[0, 1])
    print("---------------------LQG Control With Different dt---------------------------")
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}   {:10.2e} {:10.2e}" \
          .format("dt", dt_list[0], dt_list[1], dt_list[2], dt_list[3], dt_list[4], dt_list[5]))
    print( "{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}   {:10.2e} {:10.2e}"\
          .format("K1", K1_list[0],  K1_list[1],  K1_list[2],  K1_list[3],  K1_list[4], K1_list[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}   {:10.2e} {:10.2e}" \
          .format("K2", K2_list[0], K2_list[1], K2_list[2], K2_list[3], K2_list[4], K2_list[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}   {:10.2e} {:10.2e}" \
          .format("G1", G1_list[0], G1_list[1], G1_list[2], G1_list[3], G1_list[4], G1_list[4]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}   {:10.2e} {:10.2e}" \
          .format("G2", G2_list[0], G2_list[1], G2_list[2], G2_list[3], G2_list[4], G2_list[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}   {:10.2e} {:10.2e}" \
          .format("cost", cost_list[0], cost_list[1], cost_list[2], cost_list[3], cost_list[4], cost_list[5]))
    print("{:12s}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}   {:10.2e} {:10.2e}" \
          .format("testing cost", test_cost[0], test_cost[1], test_cost[2], test_cost[3], test_cost[4], test_cost[5]))

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


dt_test()