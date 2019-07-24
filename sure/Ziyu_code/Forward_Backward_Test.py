# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: June 2019
# Description: This program tests if the forward and backward Kalman filter gradient computation give the same results

import numpy as np
import Kalman_filter_SGD_forward as forward
import Kalman_filter_SGD_backward as backward

grad_diff_1 = []
grad_diff_2 = []
err_diff = []

for i in range(1000):

    K11 = 1 + np.random.rand()
    K12 = 1 + np.random.rand()
    seed = np.random.randint(2000)
    K = np.array([[K11], [K12]])

    gradF, errF = forward.compute_gradient(K, seed)

    L_state, L_obs, L_est = backward.generate_path(K, seed)
    gradB = backward.compute_gradient(K, L_state, L_obs, L_est)
    errB = backward.compute_error(L_state, L_est)

    # print(gradF - gradB)
    # print(abs(errF - errB))

    grad_diff = gradF - gradB

    grad_diff_1.append(abs(grad_diff[0][0]))
    grad_diff_2.append(abs(grad_diff[0][1]))

    err_diff.append(abs(errF - errB))

print(max(grad_diff_1))
print(max(grad_diff_2))
print(np.mean(grad_diff_1))
print(np.mean(grad_diff_2))
print(np.mean(err_diff))
