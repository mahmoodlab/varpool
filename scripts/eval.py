import pandas as pd
import os
import argparse
from time import time
import yaml

import torch
from torch.utils.data import DataLoader

from var_pool.nn.train.loops import eval_loop
from var_pool.nn.stream_evaler import ClfEvaler, DiscreteSurvivalEvaler,\
    CoxSurvivalEvaler, RankSurvivalEvaler
from var_pool.nn.datasets.BagDatasets import BagDataset
from var_pool.processing.clf_utils import dict_split_clf_df
from var_pool.processing.discr_surv_utils import dict_split_discr_surv_df
from var_pool.nn.seeds import set_seeds
from var_pool.script_utils import parse_mil_task_yaml
from var_pool.file_utils import find_fpaths, join_and_make
from var_pool.mhist.tcga_agg_slides_to_patient_level import \
    tcga_agg_slides_to_patient_level
from var_pool.mhist.get_model_from_args import get_model
from var_pool.gpu_utils import assign_free_gpus

parser = argparse.\
    ArgumentParser(description='Evaulates a trained attention MIL network.')

parser.add_argument('--task_fpath', type=str,
                    # default to task.yaml file saved in current directory
                    default='yaml/luad_vs_lusc.yaml',
                    help='Filepath to .yaml file containing the information '
                         'for  this task. It should include entries for: \n'
                         'feats_dir =  directory containing WSI bag features as .h5 files.\n'
                         'y_fpath: csv file containing the response labels for each bag and the train/val/test spilts. See code for how the csv file should be formatted.\n'
                         'task: a string indicated which kind of task we are solving. Should be one of "clf", "surv_cov" or "surv_discr"\n'
                         'train_dir: directory to where training results are saved e.g. model checkpoints and logging information.')

parser.add_argument('--name', type=str, default=None,
                    help='(Optional) Name of this experiment; the results will be saved in train_dir/name/. If name=time, will name the directory after the current date ane time.')

parser.add_argument('--seed', type=int, default=1,
                    help='Random seed for experiment reproducibility.')


parser.add_argument('--split', type=str, default='test',
                    nargs='+',
                    choices=['train', 'val', 'test'],
                    help='Which split to evaluate. If multiple are provided, will concatenate.')


parser.add_argument('--save_dir', type=str, default=None,
                    help='(Optional) Directory where to save the results. If not provided, will use the original train directory from the yaml file.')


parser.add_argument('--arch', type=str, default='amil_slim',
                    choices=['amil_slim', 'amil_nn', 'amil_var_nn', 'sum_mil', 'sum_var_mil', 'patchGCN', 'patchGCN_varpool'],
                    help="Which neural network architecture to use.\n"
                         "'amil_slim' just does attention mean pooling with a final linear layer.\n"
                         "'amil_nn' does attention mean pooling with an additional neural network layers applied to the instance embeddings and the mean pooled output.")


# architecture
parser.add_argument('--final_nonlin', default='relu', type=str,
                    choices=['relu', 'tanh', 'identity'],
                    help='Choice of final nonlinearity for architecture.')

parser.add_argument('--attn_latent_dim', default=256, type=int,
                    help='Dimension of the attention latent space.')

parser.add_argument('--freeze_var_epochs',
                    default=None, type=int,
                    help='Freeze the variance pooling weights for an initial number of epochs.')

# For everything but amil slim
parser.add_argument('--head_n_hidden_layers', default=1, type=int,
                    help='Number of hidden layers in the head network (excluding the final output layer).')


parser.add_argument('--head_hidden_dim', default=256, type=int,
                    help='Dimension of the head hidden layers.')

parser.add_argument('--dropout',
                    action='store_true', default=False,
                    help='Use dropout (p=0.25 by default).')

parser.add_argument('--mode',
                    type=str, default='patch')

# For var pool
parser.add_argument('--n_var_pools', default=100, type=int,
                    help='Number of projection vectors for variance pooling.')

parser.add_argument('--var_act_func', default='sigmoid', type=str,
                    choices=['sqrt', 'log', 'sigmoid', 'identity'],
                    help='Activation function for var pooling.')

parser.add_argument('--separate_attn',
                    action='store_true', default=False,
                    help='Use separate attention branches for the mean and variance pools.')

# For rank loss
parser.add_argument('--rank_loss_phi', default='sigmoid', type=str,
                    choices=['sigmoid', 'relu'],
                    help='The phi function for rank loss.')


args = parser.parse_args()

start_time = time()

# load task info from yaml file
feat_dir, y_fpath, train_dir, task = parse_mil_task_yaml(fpath=args.task_fpath)

########################
# Identify/Assign gpus #
########################
device = assign_free_gpus()

set_seeds(device=device, seed=args.seed)

##################################
# Setup paths for saving results #
##################################
if isinstance(args.split, str):
    split_name = args.split
else:
    split_name = '_'.join(args.split)

# where to load the trained results
train_dir = os.path.join(train_dir, args.name) \
    if args.name is not None else train_dir
checkpoint_dir = join_and_make(train_dir, 'checkpoints')
checkpoint_fpath = os.path.join(checkpoint_dir, 's_checkpoint.pt')

# where to save the evaluation results
save_dir = args.save_dir if args.save_dir is not None else train_dir
results_fpath = os.path.join(save_dir, split_name + '_results.yaml')
eval_preds_fpath = os.path.join(save_dir, split_name + '_preds')


#######################################################
# load response data along with train/va/test splits #
#######################################################

# make sure this is formatted correctly e.g. see dict_split functions below
y_df = pd.read_csv(y_fpath)

if task == 'clf':

    y_split, label_encoder = dict_split_clf_df(y_df,
                                               label_col='label',
                                               split_col='split',
                                               index_col='sample_id')
    if isinstance(args.split, str):
        y_eval = y_split[args.split]
    else:
        # concatenate multiple splits
        y_eval = pd.concat([y_split[split] for split in args.split])

    class_names = label_encoder.classes_
    n_classes = len(class_names)

elif task in ['discr_surv', 'cox_surv', 'rank_surv']:

    # TODO: eventually make separate for cox_surv/rank_surv since
    # I don't think these will necessarily have time_bin
    y_split = dict_split_discr_surv_df(y_df,
                                       time_bin_col='time_bin',
                                       censor_col='censorship',
                                       split_col='split',
                                       time_col='survival_time',
                                       index_col='sample_id')

    if isinstance(args.split, str):
        y_eval = y_split[args.split]
    else:
        # concatenate multiple splits
        y_eval = pd.concat([y_split[split] for split in args.split])

    n_time_bins = len(set(y_eval['time_bin'].values))

else:
    raise NotImplementedError("Not implemented for {}".format(task))

# samples in test set
test_samples = y_eval.index.values

######################
# setup for datasets #
######################


# file paths containing features features
# TODO: uncomment when we remove TCGA specific code below
# eval_fpaths = find_fpaths(folders=feat_dir, ext=['h5', 'pt'],
#                           names=test_samples)

# format these to the patient level
# TODO: eventually remove this from var_pool -- just used for experiments in paper!
eval_fpaths = find_fpaths(folders=feat_dir, ext=['h5', 'pt'])
eval_fpaths = tcga_agg_slides_to_patient_level(eval_fpaths,
                                               names=test_samples)


if len(eval_fpaths) == 0:
    raise RuntimeError("No test files found in {}".format(feat_dir))

# Data loading bureaucracy
dataset_kws = {}
loader_kws = {'num_workers': 4} if device.type == "cuda" else {}

# the datasets will only use fpaths with a corresponding index in y
dataset_eval = BagDataset(bag_fpaths=eval_fpaths, y=y_eval, task=task,
                          **dataset_kws)


######################
# Setup data loaders #
######################

# used to evaulate the validation metrics
loader_eval = DataLoader(dataset_eval, batch_size=1,
                         shuffle=False, drop_last=False,
                         **loader_kws)

n_instances_test, n_bag_feats = dataset_eval.get_bag_summary()


###############
# Setup Model #
###############

if task == 'clf':
    out_dim = n_classes
elif task == 'discr_surv':
    out_dim = n_time_bins
elif task == 'cox_surv':
    out_dim = 1

elif task == 'rank_surv':
    out_dim = 1

else:
    raise NotImplementedError("Not implemented for {}".format(task))


model = get_model(args=args, n_bag_feats=n_bag_feats, out_dim=out_dim)

# Load checkpoints
model.load_state_dict(torch.load(checkpoint_fpath))
model.to(device)

################
# Setup evaler #
################
if task == 'clf':
    evaler = ClfEvaler(class_names=class_names)

elif task == 'discr_surv':
    evaler = DiscreteSurvivalEvaler()

elif task == 'cox_surv':
    evaler = CoxSurvivalEvaler()

elif task == 'rank_surv':
    evaler = RankSurvivalEvaler(phi=args.rank_loss_phi)

else:
    raise NotImplementedError("Not implemented for {}".format(task))


############
# Evaluate #
############

print("Evaluating {} with {} samples".format(args.split, len(dataset_eval)))

eval_loss, eval_metrics = \
    eval_loop(model=model,
              loss_func=None,
              loader=loader_eval,
              evaler=evaler,
              device=device,
              tqdm_desc='Evaluation',
              mode=args.mode)


################
# Save results #
################
# save predictions to disk
evaler.save_tracked_data(eval_preds_fpath, sample_ids=dataset_eval.bag_names)

# save numerical results
results = {k: v for (k, v) in eval_metrics.items()}
results = {k: float(v) for (k, v) in results.items()}  # annoying numpy dtypes
print(results)
with open(results_fpath, 'w') as file:
    yaml.dump(results, file)
