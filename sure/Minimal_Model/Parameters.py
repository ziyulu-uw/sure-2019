# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: Parameters which could be used globally \
#               throughout the project are set here

import numpy as np

class time_param:
    t1 = 200
    N  = t1 + 1                 # every minute is one step
    t  = np.linspace(0, t1, N)  # time discretization

class glucose: #parameters in paper 1 minimal model
    Gb = 5         # basal plasma glucose (unit: mmole/l)
    p1 = 0.0337    # param determined by bio-experiments (unit: min^-1)
    SG = p1        # glucose effectiveness
    D  = 3/2         # administered glucose dose (unit: mmole/kg) \
                   # ????? notice: i guess the value from Fig1, but I am not sure!!!
    V  = 1.88      # parameter (unit: dl/kg)
    #R  = D/(0.1*V) # G0 (unit: mmol/l)
    R  = 6
    G0 = 6       # initial condition of glucose G(t) (unit: mmol/l)

class insulin:
    # Insulin parameters determined by bio-experiments
    p2, p3 = 0.022, 1.7e-5 # (unit: min^-1, min^-1, min^-1 mU/l)
    SI = p3 / p2  # glucose effectiveness (unit: mU/l)