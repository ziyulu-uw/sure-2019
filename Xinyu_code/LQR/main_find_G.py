# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: find G use qausi Newton Method

import numpy as np
from scipy.optimize import minimize
from cost import cost_fun
from Model_tools import S_matrix
from cost import cost_der
from EigenTest import M_eval
solution = minimize(cost_fun,(-0.01,-0.01),method = "BFGS",jac=cost_der)
M_eval(np.reshape(solution.x,[1,2]),True)
print(solution)
