# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program calls gradient testing or stochastic gradient descent

import numpy as np

###### Gradient testing ######
# Note: Manually uncomment line 15 in path_generation.py to ensure that identical paths are generated
# import gradient_tester
# K = np.array([[1.0], [1.0]])
# gradient_tester.check_order(K, 0.0001, 1, 10)

###### Stochastic gradient descent testing ######
# Note: Make sure line 15 in path_generation.py is commented out
import SGD_wrapper
import filter_tester
K = np.array([[5.0], [5.0]])
K_, err_L = SGD_wrapper.Stochastic_gradient_descent(K, 5000, 0.001, [1, 5, 10, 20, 27])
test_err = filter_tester.test(K_)