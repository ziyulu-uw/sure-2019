# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: find G use qausi Newton Method

import numpy as np
from scipy.optimize import minimize
from cost import cost_fun
from cost import cost_der
from EigenTest import M_eval

"""find G use BFGS"""
print("Find G with BFGS:")
solution1 = minimize(cost_fun,(-0.01,-0.01),method = "BFGS",jac=cost_der)
print(solution1)
M_eval(np.reshape(solution1.x,[1,2]),True)
print()

"""find G use Nelder-Mead"""
print("Find G with Nelder-Mead:")
solution2 = minimize(cost_fun,(-0.01,-0.01),method = "Nelder-Mead")
print(solution2)
print()
M_eval(np.reshape(solution2.x,[1,2]),True)
