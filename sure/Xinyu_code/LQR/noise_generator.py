# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: generate gaussian noise in the process

import numpy as np

def system_noise(R,n):
    '''generate process noise
    @parameter: covariance matrix 2*2 R
                the number of paths n
    @return: n*2*1 gaussian noise with mean 0, covariance R'''
    W = np.random.multivariate_normal(np.array([0,0]),np.real(R),n)
    W = np.reshape(W,[n,2,1])
    return W