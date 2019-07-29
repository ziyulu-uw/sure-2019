# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program is about do LQG control and Kalman Filter Estimation, to compare this gradient descent 

import numpy as np
import random
from initialization import *
# params are set here: Constant matrices: A,B,C; Cov: R,S; N-num of time step
import LQG
# simulation with LQG is done here
from LQG_tool import Plot_K, Plot_G, Plot_X, display, generate_noise
# This program offers some functions that used in LQG and plot
from LQG_loss_computation import compute_cost, Plot_Cost,after_train_cost
# cost plot and computation are done here

n = 1000  # number of paths

DoLQG = True  # If you only want to see Kalman Filter Estimation, set here to be False;
              # If you want to do LQG, set here to be True
              
# Find out Sn (a series of matrices that will be used in Gn calculation)
# Sn is determined by a matrix Riccati difference equation backward in time
Sn_list = LQG.Sn_backward(A, B, r, n, N)

# Generate process and observation noise
W = generate_noise(R, n, N+1)  # process noise: R-cov  [Wn is not used in the simulation]
V = generate_noise(S, n, N+1)  # observation noise: S-cov

# Plot K,G,and X
#Plot_K(K_val, t)
#Plot_X(X_val, t)
if DoLQG:
    # Simulate the process with LQG control
    K_val, G_val, X_val, TrueX_val = LQG.LQG_simulation(x0, v0, A, B, C, R, S, r, Sn_list, W, V, n, N, DoLQG)
display(G_val, K_val)  # display the G and K in the steady state

#Plot_G(G_val, t)
cost1 = compute_cost(X_val, TrueX_val, G_val, N, r)
print("Cost with transition Kn and Gn:   ", '{:.10e}'.format(cost1))

# get steady state K and G by taking K and G from the end
K = np.reshape(np.average(K_val, axis=0)[-1, :], [2, 1])
G = np.reshape(G_val[-1, :], [1, 2])
G_val_ = np.zeros([N+1, 2])
G_val_[:, 0] = G[0, 0]  # in this case, G_val_ from t=0 to t=N should have same G1 and G2 values
G_val_[:, 1] = G[0, 1]

# now, let simulate the path with the fixed K and G found in above simulation
test_cost = []
s_l = [1, 2, 3, 4, 5]
for k in s_l:
    random.seed(k)
    # Generate process and observation noise
    W = generate_noise(R, n, N + 1)  # process noise: R-cov  [Wn is not used in the simulation]
    V = generate_noise(S, n, N + 1)  # observation noise: S-cov
    X_val_, TrueX_val_ = after_train_cost(K, G, W, V, A, B, C, x0, v0, n, N)
    cost2       = compute_cost(X_val_, TrueX_val_, G_val_, N, r)
    test_cost.append(cost2)
avg = np.average(np.array(test_cost))
print("Cost with K and G in steady state:", '{:.10e}'.format(avg))

# Plot use transitional Kn, Gn simulation and use fixed K, G simulation
X_list1     = [X_val, X_val_]
X_list2     = [TrueX_val, TrueX_val_]
G_val_list  = [G_val, G_val_]
num_of_plot = 2
str_list    = ["transitional Kn, Gn", "steady state K, G"]
cost_list = Plot_Cost(X_list1, X_list2,  G_val_list, t, N, r, num_of_plot, str_list)
