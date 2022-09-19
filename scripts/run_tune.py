import argparse
import pathlib
import os
from functools import partial
from multiprocessing import Pool
from sklearn.model_selection import ParameterGrid
import numpy as np
import torch

from var_pool.nn.tune_utils import run_train
from var_pool.script_utils import parse_mil_task_yaml

parser = argparse.\
    ArgumentParser(description='Runs a grid serach over tuning parameters using a validation set.')

parser.add_argument('--task_fpath', type=str,
                    default='yaml/task.yaml',
                    help='Filepath to .yaml file containing the information '
                         'for  this task.')


parser.add_argument('--skip_if_exists',
                    action='store_true', default=False,
                    help='Skip running a tuning parameter setting if the results file alreay exists. Useful if tuning crashed halfway through.')


# architecture
parser.add_argument('--arch', default='amil_slim', type=str,
                    choices=['amil_slim', 'amil_nn', 'amil_var_nn', 'sum_mil', 'sum_var_mil', 'patchGCN', 'patchGCN_varpool'],
                    help="Which neural network architecture to use.")

parser.add_argument('--attn_latent_dim', default=256, type=int, nargs="+",
                    help='Dimension of the attention latent space.')

parser.add_argument('--mode', default='patch', type=str,
                    choices=['patch', 'graph'],
                    help='Framework paradigm. patch for independent path assumption. graph for contextual-aware algorithms')

# For everything but amil slim
parser.add_argument('--head_n_hidden_layers', default=1, type=int, nargs="+",
                    help='Number of hidden layers in the head network (excluding the final output layer).')


parser.add_argument('--head_hidden_dim', default=256, type=int,  nargs="+",
                    help='Dimension of the head hidden layers.')

# optimization
parser.add_argument('--n_epochs', default=100, type=int,
                    help='Number of training epochs. Note this defaults to 100.')

parser.add_argument('--grad_accum', default=1, type=int,
                    help='Number of gradient accumulation steps.')


parser.add_argument('--no_early_stopping',
                    action='store_true', default=False,
                    help='Do not use early stopping.')

parser.add_argument('--es_monitor', default='loss', type=str,
                    choices=['loss', 'metric'],
                    help='Should early stopping monitor the validation loss or another metric (e.g. c-index, auc).')

parser.add_argument('--stop_patience', default=20, type=int,
                    help='Number of patience steps for early stopping.')

parser.add_argument('--plateau_patience', default=5, type=int,
                    help='Number of patience steps for ReduceLROnPlateau learning rate scheduler.')

parser.add_argument('--plateau_factor', default=np.sqrt(.1), type=float,
                    help='The factor argument for ReduceLROnPlateau.')


parser.add_argument('--seed', type=int, default=1, nargs="+",
                    help='The random seed for training e.g. used to initialized network weights, order of shuffle, etc. Providing multiiple seeds lets you do multiple network initializations.')


# Tuning params
parser.add_argument('--batch_size', default=1, type=int, nargs="+",
                    help='Batch size for training. If this is > 1 then you must set fixed_bag_size.')


parser.add_argument('--fixed_bag_size', default=None, nargs="+",
                    help='Fix the number of instances in each bag for training.  E.g. we randomly sample a subset of instances from each bag. This can be used to speed up the training loop. To use a batch size larger than one you must set this value. By passing in fixed_bag_size=max this will automatically set fixed_bag_size to be the largest bag size of the training set.')


parser.add_argument('--n_var_pools', default=10,
                    type=int, nargs="+",
                    help='Number of projection vectors for variance pooling.')

parser.add_argument('--var_act_func',
                    default='log',
                    type=str, nargs="+",
                    help='Choice of activation functions for var pooling.')

parser.add_argument('--lr', default=2e-4,
                    type=float, nargs="+",
                    help='The learning rates to try.')

parser.add_argument('--separate_attn', default=0,
                    type=int, nargs="+", choices=[0, 1],
                    help='Separate attn flag. 0 means no separate attn, 1 means yes separate attn. If 0, 1 are provided then this will be tuned over')

parser.add_argument('--final_nonlin', default='relu',
                    type=str, nargs='+',
                    help='Final nonlinearity')

# For parallelization
parser.add_argument('--num_workers', default=3,
                    type=int,
                    help='Number of parallel workers for CV')

args = parser.parse_args()

###############################################
# Setup parameter setting to pass to train.py #
###############################################

# fixed parameters that are not tuned over
fixed_params = {'task_fpath': args.task_fpath,

                'arch': args.arch,

                'imbal_how': 'resample',

                'mode': args.mode,

                'n_epochs': args.n_epochs,
                'grad_accum': args.grad_accum,

                'lr_scheduler': 'plateau',
                'plateau_factor': args.plateau_factor,
                'plateau_patience': args.plateau_patience,
                'stop_patience': args.stop_patience,
                'es_monitor': args.es_monitor
                }


fixed_flags = ['dropout']
if not args.no_early_stopping:
    fixed_flags.append('early_stopping')

# setup parameters to be tuned over
tunable_params = ['lr', 'attn_latent_dim',
                  'fixed_bag_size', 'batch_size',
                  'seed'  # multiple inits, not actually a "tune param"!
                  ]   # every arch gets these
tunable_flags = []  # every arch gets these
if args.arch == 'amil_slim':
    pass

elif args.arch == 'amil_nn':
    tunable_params.extend(['head_n_hidden_layers', 'head_hidden_dim'])

elif args.arch == 'patchGCN':
    tunable_params.extend(['head_n_hidden_layers', 'head_hidden_dim'])

elif args.arch == 'amil_var_nn':
    tunable_params.extend(['head_n_hidden_layers', 'head_hidden_dim',
                           'var_act_func', 'n_var_pools'])

    tunable_flags.append('separate_attn')

elif args.arch == 'sum_var_mil':
    tunable_params.extend(['head_n_hidden_layers', 'head_hidden_dim',
                           'var_act_func', 'n_var_pools'])

    tunable_flags.append('separate_attn')

else:
    pass

################################
# Process tunable params/flags #
################################

# pull out tuning parameter grids from argparse
param_grid = {}
for param_name in tunable_params:
    # pull out param settings from args
    param_settings = args.__dict__[param_name]

    # if this is None then don't add it
    if param_settings is None:
        continue

    if isinstance(param_settings, list):
        # if multiple parameters were provided we tune over them
        param_grid[param_name] = param_settings
    else:
        # otherwise these are just fixed parameters
        fixed_params[param_name] = param_settings

# pull out tunable flags from argparse
# format them too bool
for flag_name in tunable_flags:
    # pull out param settings from args
    flag_settings = args.__dict__[flag_name]

    if isinstance(flag_settings, list):
        # if multiple parameters were provided we tune over them
        param_grid[flag_name] = [bool(int(f)) for f in flag_settings]
    else:
        # otherwise these are just fixed parameters
        flag_on = bool(int(flag_settings))
        if flag_on:
            fixed_flags.append(flag_name)


#############################
# Run setup tuning proceses #
#############################

# count availabe cuda devices
n_devices = torch.cuda.device_count()

# load task info from yaml file
feat_dir, y_fpath, train_dir, task = parse_mil_task_yaml(fpath=args.task_fpath)

# get path to train.py -- should be in same directory as this script
script_dir = pathlib.Path(__file__).parent.resolve()
train_script_fpath = os.path.join(script_dir, 'train.py')


pool = Pool(processes=args.num_workers)

# set the fixed arguments for run_train
# this is callable(tune_params, tune_idx, device)
run_func = partial(run_train,
                   script_fpath=train_script_fpath,
                   train_dir=train_dir,
                   fixed_params=fixed_params,
                   fixed_flags=fixed_flags,
                   skip_if_results_exist=args.skip_if_exists)

# list of dicts of tune param settings
tune_seq = list(ParameterGrid(param_grid))
n_tune = len(tune_seq)
tune_idxs = range(n_tune)

print("Tuning over {} settings".format(n_tune))

# which device each task goes to
device_list = [idx % n_devices for idx in range(n_tune)]

# each entry is (tune_params, tune_idx, device)
run_args = list(zip(tune_seq, tune_idxs, device_list))

##############
# Run tuning #
##############

pool.starmap(run_func, run_args)

pool.close()
pool.join()
