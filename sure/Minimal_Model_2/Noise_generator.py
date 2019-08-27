# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Generate noise for the whole run

import numpy as np

def noise_path(init_cond, total_N_meas, seed_num=10):
    """
     create all the noise that will be used in a simulation.
    @:param  init_cond:             G0,X0,I0,Ra0
    @:param  total_N_meas:          Number of measurements of the whole run
    @:param  seed_num:              a number to set the seed
    @:return process_noise:        [4*N_meas], the covariance value is set to be 1/5 of the initial condition, all 4 state has process noise
             observation_noise:    [1*N_meas], the covariance is set to be 1/20 of G0
    """
    np.random.seed(seed_num)
    G0, X0, I0, Ra0 = init_cond
    process_noise_cov = 1/10 * np.array(init_cond)

    process_noise = np.zeros([4, total_N_meas])
    for i in range(len(init_cond)):
        process_noise[i,:] =  np.random.normal(0, process_noise_cov[i], [1,total_N_meas])
    observation_noise_cov = 1/20 * G0
    observation_noise = np.random.normal(0,observation_noise_cov,[1, total_N_meas])

    return process_noise, observation_noise


