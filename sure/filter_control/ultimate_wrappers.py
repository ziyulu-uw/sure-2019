# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program implements a wrapper function for the optimization algorithms \
# which plots the results and the comparison between theoretical results

import ultimate_optim
import stability_check
import performance_test
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


def wrapper(X0, A, C, B, G0, K0, N, S, R, d_X, d_Z, d_U, r, n, act, L, g, s_l, alpha, betas, momentum, M, comp, avg, which, zoom):
    # X0 -- initial state, A -- state transition matrix, C -- observation matrix, \
    # B -- control coefficient matrix, G0 -- initial control gain, K0 -- initial Kalman gain, \
    # N -- number of total time steps, S -- observation noise covariance matrix, R -- system noise covariance matrix, \
    # d_X -- dimension of state, d_Z -- dimension of observation, d_U -- dimension of control, \
    # r -- scaling factor, n -- number of total gradient steps, \
    # act -- set to True to activate learning rate scheduler, \
    # L -- list of milestones, g -- multiplicative factor of learning rate decay,  s_l -- a list of random seeds, \
    # alpha -- learning rate, \
    # betas -- coefficients for computing running average (set to None to use the default values), \
    # momentum -- momentum factor (can set to 0), M -- minibatch size, \
    # avg -- if not None, the final result is averaged over last avg steps, \
    # comp -- an array containing the theoretical optimal K, G, and cost, \
    # which -- name of the optimization algorithm to use (SGD, Adam, or RMSprop), \
    # zoom -- to plot the result [zoom: ]; if None, plot the whole training process
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
                                                                        g, alpha, betas, momentum, M, s, avg, comp=None, which=which)
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
        plt.grid(True)
        plt.show()

        return K_avg, G_avg, F_avg

    else:
        diff_K_avg = np.zeros(n)
        diff_G_avg = np.zeros(n)
        K1_avg = np.zeros(n)
        K2_avg = np.zeros(n)
        G1_avg = np.zeros(n)
        G2_avg = np.zeros(n)
        print("---- After {} iterations ----".format(n))
        print("seed      K1            K2         G1         G2       InitialLoss   FinalLoss   Initial_K_Diff  Final_K_Diff  Intial_G_Diff  Final_G_Diff  Max_K_Gradient  Max_G_Gradient")
        for s in s_l:
            try:
                K, G, F_l, grad_K_l, grad_G_l, diff_K_l, diff_G_l, K1_l, K2_l, G1_l, G2_l = ultimate_optim.optimize(X0, A, C, B, G0, K0, N, S, R, d_X, d_Z, d_U, r, n, act, L,
                                                                                       g, alpha, betas, momentum, M, s, avg, comp, which)
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
            K1_avg += K1_l
            K2_avg += K2_l
            G1_avg += G1_l
            G2_avg += G2_l

        K_avg = K_avg / len(s_l)
        G_avg = G_avg / len(s_l)
        F_avg = F_avg / len(s_l)
        diff_K_avg = diff_K_avg / len(s_l)
        diff_G_avg = diff_G_avg / len(s_l)
        K1_avg = K1_avg / len(s_l)
        K2_avg = K2_avg / len(s_l)
        G1_avg = G1_avg / len(s_l)
        G2_avg = G2_avg / len(s_l)

        print("---- Averaging over {} random seeds ----".format(len(s_l)))
        print("K11 is{:10.2e}, K12 is{:10.2e}".format(K_avg[0][0], K_avg[1][0]))
        print("G1 is{:10.2e}, G2 is{:10.2e}".format(G_avg[0][0], G_avg[0][1]))
        if avg is None:
            print("The final loss is{:10.2e}".format(F_avg[-1]))
            print("The norm of difference between final K and theoretical optimal K is{:10.2e}".format(diff_K_avg[-1]))
            print("The norm of difference between final G and theoretical optimal G is{:10.2e}".format(diff_G_avg[-1]))
        if avg is not None:
            final_K_diff = LA.norm(K_avg - comp[0])
            final_G_diff = LA.norm(G_avg - comp[1])
            final_loss = performance_test.test(X0, A, C, B, G_avg, K_avg, N, R, S, r, d_X, d_Z, d_U, M, disp=False)[0][0]
            print("The final loss is{:10.2e}".format(final_loss))
            print("The norm of difference between final K and theoretical optimal K is{:10.2e}".format(final_K_diff))
            print("The norm of difference between final G and theoretical optimal K is{:10.2e}".format(final_G_diff))

        x = [i for i in range(n)]

        plt.figure(1)
        best_cost = np.ones(n) * comp[2]  # comp[2] is the theoretical optimal cost
        if zoom is None:
            plt.plot(x, F_avg, label='SGD cost')
            plt.plot(x, best_cost, label='theoretical minimal cost')
            plt.yscale("log")
        else:
            plt.plot(x[n-zoom:], F_avg[n-zoom:], label='SGD cost')
            plt.plot(x[n-zoom:], best_cost[n-zoom:], label='theoretical minimal cost')

        plt.rcParams["axes.titlesize"] = 8
        # plt.title("{} algorithm with {} steps and step size {}, averaged over {} random seeds". \
        #           format(which, str(n), str(alpha), str(len(s_l))))
        plt.title("{} algorithm with {} steps, step size {}, minibatch size {}, smoothing constant {}". \
                  format(which, str(n), str(alpha), str(M), str(betas)))
        plt.xlabel("number of optimization steps")
        plt.ylabel("mean squared error of one simulation")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(2)
        if zoom is None:
            plt.plot(x, diff_K_avg, label='K')
            plt.plot(x, diff_G_avg, label='G')
        else:
            plt.plot(x[n-zoom:], diff_K_avg[n-zoom:], label='K')
            plt.plot(x[n-zoom:], diff_G_avg[n-zoom:], label='G')
        # plt.yscale("log")
        plt.rcParams["axes.titlesize"] = 8
        # plt.title("{} algorithm with {} steps and step size {}, averaged over {} random seeds". \
        #           format(which, str(n), str(alpha), str(len(s_l))))
        plt.title("{} algorithm with {} steps, step size {}, minibatch size {}, smoothing constant {}". \
                  format(which, str(n), str(alpha), str(M), str(betas)))
        plt.xlabel("number of optimization steps")
        plt.ylabel("norm of difference between current K, G and theoretical optimal K, G", fontsize=8)
        plt.legend()
        plt.grid(True)
        plt.show()

        ''''
        plt.figure(3)
        if zoom is None:
            plt.plot(x, K1_avg, label='K1')
            plt.plot(x, K2_avg, label='K2')
        else:
            plt.plot(x[n - zoom:], K1_avg[n - zoom:], label='K1')
            plt.plot(x[n - zoom:], K2_avg[n - zoom:], label='K2')
        # plt.yscale("log")
        plt.rcParams["axes.titlesize"] = 8
        # plt.title("{} algorithm with {} steps and step size {}, averaged over {} random seeds". \
        #           format(which, str(n), str(alpha), str(len(s_l))))
        plt.title("{} algorithm with {} steps, step size {}, minibatch size {}, smoothing constant {}". \
                  format(which, str(n), str(alpha), str(M), str(betas)))
        plt.xlabel("number of optimization steps")
        plt.ylabel("norm of difference between current K1, K2 and theoretical optimal K1, K2", fontsize=8)
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(4)
        if zoom is None:
            plt.plot(x, G1_avg, label='G1')
            plt.plot(x, G2_avg, label='G2')
        else:
            plt.plot(x[n - zoom:], G1_avg[n - zoom:], label='G1')
            plt.plot(x[n - zoom:], G2_avg[n - zoom:], label='G2')
        # plt.yscale("log")
        plt.rcParams["axes.titlesize"] = 8
        # plt.title("{} algorithm with {} steps and step size {}, averaged over {} random seeds". \
        #           format(which, str(n), str(alpha), str(len(s_l))))
        plt.title("{} algorithm with {} steps, step size {}, minibatch size {}, smoothing constant {}". \
                  format(which, str(n), str(alpha), str(M), str(betas)))
        plt.xlabel("number of optimization steps")
        plt.ylabel("norm of difference between current G1, G2 and theoretical optimal G1, G2", fontsize=8)
        plt.legend()
        plt.grid(True)
        plt.show()
        '''
        plt.figure(4)
        if zoom is None:
            plt.plot(x, K1_avg, label='K1')
            plt.plot(x, K2_avg, label='K2')
            plt.plot(x, G1_avg, label='G1')
            plt.plot(x, G2_avg, label='G2')
        else:
            plt.plot(x[n - zoom:], K1_avg[n - zoom:], label='K1')
            plt.plot(x[n - zoom:], K2_avg[n - zoom:], label='K2')
            plt.plot(x[n - zoom:], G1_avg[n - zoom:], label='G1')
            plt.plot(x[n - zoom:], G2_avg[n - zoom:], label='G2')
        # plt.yscale("log")
        plt.rcParams["axes.titlesize"] = 8
        # plt.title("{} algorithm with {} steps and step size {}, averaged over {} random seeds". \
        #           format(which, str(n), str(alpha), str(len(s_l))))
        plt.title("{} algorithm with {} steps, step size {}, minibatch size {}, smoothing constant {}". \
                  format(which, str(n), str(alpha), str(M), str(betas)))
        plt.xlabel("number of optimization steps")
        plt.ylabel("difference between current K, G and theoretical optimal K, G", fontsize=8)
        plt.legend()
        plt.grid(True)
        plt.show()


        return K_avg, G_avg, F_avg, diff_K_avg, diff_K_avg
