# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: find G use qausi Newton Method

import numpy as np
from scipy.optimize import minimize
from cost import cost_fun
from Model_tools import S_matrix
from cost import cost_der

G = np.array([-0.05,-0.2]) #initial guess
solution = minimize(cost_fun,(0.01,-0.01),method = "Nelder-Mead",jac=cost_der)
print(solution)