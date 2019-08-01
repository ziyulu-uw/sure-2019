# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program implements a wrapper function for the optimization algorithms \
# which plots the results and the comparison between theoretical results

import ultimate_optim
import stability_check
import numpy as np
import matplotlib.pyplot as plt


def wrapper(X0, A, C, B, G0, K0, N, S, R, d_X, d_Z, d_U, r, n, act, L, g, s_l, alpha, momentum, M, comp, which):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G0 -- initial control gain, K0 -- initial Kalman gain, \
    # N -- number of total time steps, S -- observation noise covariance matrix, R -- system noise covariance matrix, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control, \
    # r -- scaling factor, n -- number of total gradient steps, \
    # act -- set to True to activate learning rate scheduler, \
    # L -- list of milestones, g -- multiplicative factor of learning rate decay,  s_l -- a list of random seeds, \
    # alpha -- learning rate, momentum -- momentum factor (can set to 0), M -- minibatch size, \
    # comp -- an array containing the theoretical optimal K, G, and cost, \
    # which -- name of the optimization algorithm to use (SGD, Adam, or RMSprop)
    # a wrapper function that calls the optimize in ultimate_optim.py for s in s_l

    is_stable = stability_check.check_stability(A, B, C, K0, G0, d_X)
    if is_stable is False:
        print("Dynamics is unstable. Choose another K0 and G0")
        return
    print("Dynamics is stable")

    print("Optimizing using {} algorithm".format(which))
    print("---- Initialization ----")
    print("K11 is {}, K12 is {}".format(K0[0][0], K0[1][0]))
    print("G1 is {}, G2 is {}".format(G0[0][0], G0[0][1]))
    K_avg = np.zeros((2, 1))
    G_avg = np.zeros((1, 2))
    F_avg = np.zeros(n)

    # not comparing with theoretical result
    if comp is None:
        print("---- After {} iterations ----".format(n))
        print("seed      K1            K2         G1         G2       InitialLoss   FinalLoss   Max_K_Gradient  Max_G_Gradient")
        for s in s_l:
            try:
                K, G, F_l, grad_K_l, grad_G_l = ultimate_optim.optimize(X0, A, C, B, G0, K0, N, S, R, d_X, d_Z, d_U, r, n, act, L,
                                                                        g, alpha, momentum, M, s, comp=None, which=which)
            except TypeError:
                return

            print("{:2d}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}   {:10.2e}     {:10.2e}     {:10.2e}"\
                  .format(s, K[0][0], K[1][0], G[0][0], G[0][1], F_l[0], F_l[-1], np.amax(grad_K_l), np.amax(grad_G_l)))
            K_avg += K
            G_avg += G
            F_avg += F_l

        K_avg = K_avg/len(s_l)
        G_avg = G_avg/len(s_l)
        F_avg = F_avg/len(s_l)

        print("---- Averaging over {} random seeds ----".format(len(s_l)))
        print("K11 is{:10.2e}, K12 is{:10.2e}".format(K_avg[0][0], K_avg[1][0]))
        print("G1 is{:10.2e}, G2 is{:10.2e}".format(G_avg[0][0], G_avg[0][1]))
        print("The final loss is{:10.2e}".format(F_avg[-1]))

        x = [i for i in range(n)]
        plt.plot(x, F_avg)
        plt.yscale("log")
        plt.rcParams["axes.titlesize"] = 8
        plt.title("RMSprop algorithm with {} steps and step size {}, averaged over {} random seeds".\
                  format(str(n),str(alpha),str(len(s_l))))
        plt.xlabel("number of optimization steps")
        plt.ylabel("mean squared error of one simulation")
        plt.show()

        return K_avg, G_avg, F_avg

    else:
        diff_K_avg = np.zeros(n)
        diff_G_avg = np.zeros(n)
        print("---- After {} iterations ----".format(n))
        print("seed      K1            K2         G1         G2       InitialLoss   FinalLoss   Initial_K_Diff  Final_K_Diff  Intial_G_Diff  Final_G_Diff  Max_K_Gradient  Max_G_Gradient")
        for s in s_l:
            try:
                K, G, F_l, grad_K_l, grad_G_l, diff_K_l, diff_G_l = ultimate_optim.optimize(X0, A, C, B, G0, K0, N, S, R, d_X, d_Z, d_U, r, n, act, L,
                                                                                       g, alpha, momentum, M, s, comp, which)
            except TypeError:
                return

            print("{:2d}    {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}  {:10.2e}   {:10.2e}     {:10.2e}     {:10.2e}     {:10.2e}     {:10.2e}     {:10.2e}     {:10.2e}" \
                  .format(s, K[0][0], K[1][0], G[0][0], G[0][1], F_l[0], F_l[-1],
                          diff_K_l[0], diff_K_l[-1], diff_G_l[0], diff_G_l[-1], np.amax(grad_K_l), np.amax(grad_G_l)))
            K_avg += K
            G_avg += G
            F_avg += F_l
            diff_K_avg += diff_K_l
            diff_G_avg += diff_G_l

        K_avg = K_avg / len(s_l)
        G_avg = G_avg / len(s_l)
        F_avg = F_avg / len(s_l)
        diff_K_avg = diff_K_avg / len(s_l)
        diff_G_avg = diff_G_avg / len(s_l)

        print("---- Averaging over {} random seeds ----".format(len(s_l)))
        print("K11 is{:10.2e}, K12 is{:10.2e}".format(K_avg[0][0], K_avg[1][0]))
        print("G1 is{:10.2e}, G2 is{:10.2e}".format(G_avg[0][0], G_avg[0][1]))
        print("The final loss is{:10.2e}".format(F_avg[-1]))
        print("The norm of difference between final K and theoretical optimal K is{:10.2e}".format(diff_K_avg[-1]))
        print("The norm of difference between final G and theoretical optimal G is{:10.2e}".format(diff_G_avg[-1]))

        x = [i for i in range(n)]
        plt.figure(1)
        plt.plot(x, F_avg, label='SGD cost')
        best_cost = np.ones(n)*comp[2]  # comp[2] is the theoretical optimal cost
        plt.plot(x, best_cost, label='theoretical minimal cost')
        plt.yscale("log")
        plt.rcParams["axes.titlesize"] = 8
        plt.title("{} algorithm with {} steps and step size {}, averaged over {} random seeds". \
                  format(which, str(n), str(alpha), str(len(s_l))))
        plt.xlabel("number of optimization steps")
        plt.ylabel("mean squared error of one simulation")
        plt.legend()
        plt.show()

        plt.figure(2)
        plt.plot(x, diff_K_avg, label='K')
        plt.plot(x, diff_G_avg, label='G')
        # plt.yscale("log")
        plt.rcParams["axes.titlesize"] = 8
        plt.title("{} algorithm with {} steps and step size {}, averaged over {} random seeds". \
                  format(which, str(n), str(alpha), str(len(s_l))))
        plt.xlabel("number of optimization steps")
        plt.ylabel("norm of difference between current K, G and theoretical optimal K, G", fontsize=8)
        plt.legend()
        plt.show()

        return K_avg, G_avg, F_avg, diff_K_avg, diff_K_avg
