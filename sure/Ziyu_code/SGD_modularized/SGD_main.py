# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program calls the stochastic gradient descent wrapper

import numpy as np

# Note: Make sure line 15 in path_generation.py is commented out
import SGD_wrapper
import filter_tester
K = np.array([[5.0], [5.0]])
K_, err_L = SGD_wrapper.Stochastic_gradient_descent(K, 5000, 0.001, [1, 5, 10, 20, 27])
test_err = filter_tester.test(K_)