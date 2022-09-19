"""
Simple idea from Richard. Extract mean embeddings from WSI and train linear Cox
"""
import pandas as pd
import os
import argparse
from time import time
import numpy as np

import torch

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from var_pool.nn.datasets.BagDatasets import BagDataset
from var_pool.script_utils import parse_mil_task_yaml
from var_pool.file_utils import find_fpaths, join_and_make
from var_pool.processing.discr_surv_utils import dict_split_discr_surv_df

parser = argparse.\
    ArgumentParser(description='Trains an attention MIL network for a supervised learning task.')

parser.add_argument('--task_fpath', type=str,
                    # default to task.yaml file saved in current directory
                    default='yaml/luad_vs_lusc.yaml',
                    help='Filepath to .yaml file containing the information '
                         'for  this task. It should include entries for: \n'
                         'feats_dir =  directory containing WSI bag features as .h5 files.\n'
                         'y_fpath: csv file containing the response labels for each bag and the train/val/test spilts. See code for how the csv file should be formatted.\n'
                         'task: a string indicated which kind of task we are solving. Should be one of "clf", "surv_cov" or "surv_discr"\n'
                         'train_dir: directory to where training results are saved e.g. model checkpoints and logging information.')

parser.add_argument('--alpha', type=float, default=10,
                    help="Penalty hyperparamter for Ridge regression")

args = parser.parse_args()

start_time = time()

# load task info from yaml file
feat_dir, y_fpath, train_dir, task = parse_mil_task_yaml(fpath=args.task_fpath)

# TODO: set this working below
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# print("Training with device {} ({} cuda devices available)".
#       format(device, torch.cuda.device_count()))

##################################
# Setup paths for saving results #
##################################
train_dir = os.path.join(train_dir, 'baseline_cox')

# training logs, model checkpoints, final results file
log_dir = join_and_make(train_dir, 'log')
checkpoint_dir = join_and_make(train_dir, 'checkpoints')
summary_fpath = os.path.join(train_dir, 'summary.txt')
results_fpath = os.path.join(train_dir, 'results.yaml')
train_preds_fpath = os.path.join(train_dir, 'train_preds')
val_preds_fpath = os.path.join(train_dir, 'train_preds')


#######################################################
# load response data along with train/va/test splits #
#######################################################

# make sure this is formatted correctly e.g. see dict_split functions below
y_df = pd.read_csv(y_fpath)
y_split = dict_split_discr_surv_df(y_df,
                                   time_bin_col='time_bin',
                                   censor_col='censorship',
                                   split_col='split',
                                   time_col='survival_time',
                                   index_col='sample_id')

y_train = y_split['train']
y_val = y_split['val']
n_time_bins = len(set(y_train['time_bin'].values))


# samples in train/test sets
train_samples = y_train.index.values
val_samples = y_val.index.values

######################
# setup for datasets #
######################

# file paths containing features features
train_fpaths = find_fpaths(folders=feat_dir, ext=['h5', 'pt'],
                           names=train_samples)
val_fpaths = find_fpaths(folders=feat_dir, ext=['h5', 'pt'],
                         names=val_samples)

if len(train_fpaths) == 0:
    raise RuntimeError("No training files found in {}".format(feat_dir))
if len(val_fpaths) == 0:
    # TODO: maybe warn?
    raise RuntimeError("No validation files found in {}".format(feat_dir))

dataset_kws = {}
loader_kws = {'num_workers': 4} if device.type == "cuda" else {}

# the datasets will only use fpaths with a corresponding index in y
dataset_train = BagDataset(fpaths=train_fpaths, y=y_train, task=task,
                           **dataset_kws)
dataset_val = BagDataset(fpaths=val_fpaths, y=y_val, task=task,
                         **dataset_kws)

#######################
# Train linear Cox PH #
#######################
print("======================")
print("Training linear Cox PH")
mean_embeddings_train = []
out_train = []
for idx in range(len(dataset_train)):
    bag, y = dataset_train[idx]
    bag_embedding = torch.mean(bag, dim=0).reshape(1, -1)
    mean_embeddings_train.append(bag_embedding)
    if task == 'discr_surv':
        out_train.append((y[1], y[2]))  # (censor status, time of event)
    else:
        out_train.append((y[0], y[1]))

mean_embeddings_train = torch.cat(mean_embeddings_train, dim=0)
out_train = np.array(out_train)
out_train = Surv.from_arrays(event=out_train[:, 0], time=out_train[:, 1])

# Feed meean_embeddings into training cox model
ph = CoxPHSurvivalAnalysis(alpha=args.alpha)
ph.fit(mean_embeddings_train, out_train)

train_c_index = ph.score(mean_embeddings_train, out_train)
print("Train c-index {}".format(train_c_index))

print("======================")
print("Validating Linear Cox PH")
# Check with val data
mean_embeddings_val = []
out_val = []
for idx in range(len(dataset_val)):
    bag, y = dataset_val[idx]
    bag_embedding = torch.mean(bag, dim=0).reshape(1, -1)
    mean_embeddings_val.append(bag_embedding)
    if task == 'discr_surv':
        out_val.append((y[1], y[2]))  # (censor status, time of event)
    else:
        out_val.append((y[0], y[1]))

mean_embeddings_val = torch.cat(mean_embeddings_val, dim=0)
out_val = np.array(out_val)
out_val = Surv.from_arrays(event=out_val[:, 0], time=out_val[:, 1])

val_c_index = ph.score(mean_embeddings_val, out_val)
print("Val c-index {}".format(val_c_index))
