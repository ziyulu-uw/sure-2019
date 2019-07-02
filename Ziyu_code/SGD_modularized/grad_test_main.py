# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: July 2019
# Description: This program calls the gradient tester

import numpy as np

# Note: Manually uncomment line 15 in path_generation.py to ensure that identical paths are generated
import gradient_tester
K = np.array([[1.0], [1.0]])
gradient_tester.check_order(K, 1e-6, 1, 10)