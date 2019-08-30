from wrapper import optim_wrapper
from plotter import multi_plot
from Initialization import init_cond, N, N_meas, param_list, Gb, Ib, T, T_list, meal_params, total_t_list


Filter = [0.01, 0.01, 0.01, 0.01]
control_gain = [15, 3, 0.1, 0.5]
Filter, control_gain, cost_l, filter_l, control_l, gradF_l, gradC_l = optim_wrapper(init_cond, param_list, control_gain, Filter, Gb, Ib, N_meas, T, T_list, N,
            meal_params, which='RMSprop', alpha=1e-3, momentum=0, beta1=0.9, beta2=0.99, M=1, n=10, fname='out')

multi_plot(cost_l, p=2, which='RMSprop', nIter=10, alpha=1e-3, M=1, betas=0.9, log=True, label=None)
multi_plot(filter_l, p=1, which='RMSprop', nIter=10, alpha=1e-3, M=1, betas=0.9, log=False, label='K')
multi_plot(control_l, p=1, which='RMSprop', nIter=10, alpha=1e-3, M=1, betas=0.9, log=False, label='H')
multi_plot(gradF_l, p=1, which='RMSprop', nIter=10, alpha=1e-3, M=1, betas=0.9, log=False, label='grad K')
multi_plot(gradC_l, p=1, which='RMSprop', nIter=10, alpha=1e-3, M=1, betas=0.9, log=False, label='grad H')


'''
from wrapper import optim_wrapper
from plotter import multi_plot
Filter = [0.01, 0.01, 0.01, 0.01]
control_gain = [15, 3, 0.1, 0.5]
control_l, cost_l, grad_l = optim_wrapper(init_cond, param_list, control_gain, Filter, Gb, Ib, N_meas, T, T_list, N,
            meal_params, which='RMSprop', alpha=1e-3, momentum=0, beta1=0.9, beta2=0.99, M=1, n=10, fname=None)
multi_plot(cost_l, p=2, which='RMSprop', nIter=1000, alpha=1e-5, M=1, betas=0.99, log=True, label=None)
multi_plot(control_l, p=1, which='RMSprop', nIter=1000, alpha=1e-5, M=1, betas=0.99, log=False, label='H')
multi_plot(grad_l, p=1, which='RMSprop', nIter=1000, alpha=1e-5, M=1, betas=0.99, log=False, label='grad H')
'''