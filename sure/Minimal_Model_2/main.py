# Author: Ziyu Lu
# Email: zl1546@nyu.edu
# Date: August 2019
# Description: parses user input to run the model

import argparse
from Initialization import total_t_list, N_meas, init_cond, Gb, Ib, meal_params, param_list, T, T_list, N
import numpy as np
from wrapper import optim_wrapper

parser = argparse.ArgumentParser()

parser.add_argument('-k1', type=float, default=0.76, help='1st parameter in the filter')
parser.add_argument('-k2', type=float, default=0.000001, help='2nd parameter in the filter')
parser.add_argument('-k3', type=float, default=0.000001, help='3rd parameter in the filter')
parser.add_argument('-k4', type=float, default=0.56, help='4th parameter in the filter')
parser.add_argument('-h1', type=float, default=1.3,  help='1st parameter in the control')
parser.add_argument('-h2', type=float, default=2,  help='2nd parameter in the control')
parser.add_argument('-h3', type=float, default=0.9,  help='3rd parameter in the control')
parser.add_argument('-h4', type=float, default=0.03, help='4th parameter in the control')

parser.add_argument('-lam', '--lamda',       type=float, default=0,         help='scaling factor of control in the cost')
parser.add_argument('-w',                     type=str,   default='RMSprop', help='which optimization algorithm to use')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2,      help='learning rate')
parser.add_argument('-mon', '--momentum',     type=float, default=0,         help='momentum')
parser.add_argument('-b1', '--beta1',         type=float, default=0.9,       help='smoothing constant 1')
parser.add_argument('-b2', '--beta2',         type=float, default=0.999,     help='smoothing constant 2 (for Adam)')
parser.add_argument('-M', '--minibatch',      type=int,   default=1,         help='minibatch size')
parser.add_argument('-n', '--nIter',          type=int,   default=100,       help='number of training iterations')
parser.add_argument('-f', '--file_name',      type=str,   default='out',     help='name of the output file')


args = parser.parse_args()

np.random.seed(1)
Filter = [args.k1, args.k2, args.k3, args.k4]
control_gain = [args.h1, args.h2, args.h3, args.h4]

Filter, control_gain, cost_l, filter_l, control_l, gradF_l, gradC_l = optim_wrapper(init_cond, param_list, control_gain, Filter, Gb, Ib, N_meas, T, T_list, N,
                                                                                    meal_params, lbda=args.lamda, which=args.w, alpha=args.learning_rate, momentum=args.momentum,
                                                                                    beta1=args.beta1, beta2=args.beta2, M=args.minibatch, n=args.nIter, fname=args.file_name)
