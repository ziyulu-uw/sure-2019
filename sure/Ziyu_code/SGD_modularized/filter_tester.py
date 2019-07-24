# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program tests the performance of the optimal Kalman gain found by SGD

import path_generation as pgen
import gradient_error_computation as comp


def test(K):
    # K -- optimal Kalman gain
    # tests the performance of K on 10 random Kalman filtering problems
    # returns the average error
    # this error should be approximately equal to or slightly larger than the final loss of SGD

    avg_err = 0
    for i in range(10):
        L_state, L_obs, L_est = pgen.generate_path(K,0)
        err = comp.compute_error(L_state, L_est)
        avg_err += err

    avg_err = avg_err/10
    print("Testing result: {:.3f}".format(avg_err))

    return avg_err
