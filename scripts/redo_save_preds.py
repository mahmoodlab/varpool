"""
This script trains a multi-instance learning network to solve a supervised task e.g. classification or survival prediction.
"""
import pandas as pd
from tqdm import tqdm
import os
import argparse
from time import time
from datetime import datetime
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader as GraphDataLoader

from var_pool.nn.NLLSurvLoss import NLLSurvLoss
from var_pool.nn.CoxLoss import CoxLoss
from var_pool.nn.SurvRankingLoss import SurvRankingLoss
from var_pool.nn.train.loops import train_loop, eval_loop
from var_pool.nn.stream_evaler import ClfEvaler, DiscreteSurvivalEvaler,\
    CoxSurvivalEvaler, RankSurvivalEvaler
from var_pool.nn.datasets.BagDatasets import BagDataset
from var_pool.nn.datasets.GraphDatasets import GraphDataset
from var_pool.nn.datasets.fixed_bag_size import get_collate_fixed_bag_size
from var_pool.nn.train.EarlyStopper import EarlyStopper
# from var_pool.nn.ComparablePairSampler import ComparablePairSampler
from var_pool.nn.seeds import set_seeds
from var_pool.script_utils import parse_mil_task_yaml, write_training_summary,\
    descr_stats
from var_pool.file_utils import find_fpaths, join_and_make
from var_pool.nn.utils import get_optim, initialize_weights
from var_pool.processing.clf_utils import dict_split_clf_df, \
    get_weights_for_balanced_clf
from var_pool.processing.discr_surv_utils import dict_split_discr_surv_df, \
    get_weights_for_balanced_binXcensor
from var_pool.utils import get_counts_and_props
from var_pool.nn.utils import get_network_summary
from var_pool.gpu_utils import assign_free_gpus

from var_pool.mhist.tcga_agg_slides_to_patient_level import \
    tcga_agg_slides_to_patient_level
from var_pool.mhist.get_model_from_args import get_model

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

parser.add_argument('--name', type=str, default=None,
                    help='(Optional) Name of this experiment; the results will be saved in train_dir/name/. If name=time, will name the directory after the current date ane time.')

parser.add_argument('--seed', type=int, default=1,
                    help='The random seed for training e.g. used to initialized network weights, order of shuffle, etc.')

parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'sgd'],
                    help="Which optimization algorithm to use.")

parser.add_argument('--n_epochs', default=20, type=int,
                    help='Number of training epochs.')

parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training. If this is > 1 then you must set fixed_bag_size.')

parser.add_argument('--fixed_bag_size', default=None,
                    help='Fix the number of instances in each bag for training.  E.g. we randomly sample a subset of instances from each bag. This can be used to speed up the training loop. To use a batch size larger than one you must set this value. By passing in fixed_bag_size=max this will automatically set fixed_bag_size to be the largest bag size of the training set. If fixed_bag_size=q75, q90,... the fixed bag size will be the corresponding quantile of the trianing bag sizes.')

parser.add_argument('--lr', default=1e-3, type=float,
                    help='The learning rate.')

parser.add_argument('--lr_scheduler', default=None, type=str,
                    choices=['plateau', None],  # TODO: 'cosine', 'cyclic'
                    help="The learning rate scheduler."
                         "'plateau' means reduce on val loss plateau")

parser.add_argument('--plateau_patience', default=5, type=int,
                    help='Number of patience steps for ReduceLROnPlateau learning rate scheduler.')

parser.add_argument('--plateau_factor', default=np.sqrt(.1), type=float,
                    help='The factor argument for ReduceLROnPlateau.')

parser.add_argument('--weight_decay', default=1e-5, type=float,
                    help='Optimizer weight decay.')

parser.add_argument('--grad_accum', default=1, type=int,
                    help='Number of gradient accumulation steps.')

parser.add_argument('--early_stopping',
                    action='store_true', default=False,
                    help='Stop training early if a validation set score stops improving. If early stopping is enabled the best model (according to validation score) is checkpointed i.e. the final model is not necssarily the last epoch. If early stopping is not enabled the checkpointed model will be the model from the final epoch.')

parser.add_argument('--stop_patience', default=20, type=int,
                    help='Number of patience steps for early stopping.')

parser.add_argument('--es_monitor', default='loss', type=str,
                    choices=['loss', 'metric'],
                    help='Should early stopping monitor the validation loss or another metric (e.g. c-index, auc).')

parser.add_argument('--imbal_how', default=None, type=str,
                    choices=['resample', 'loss_weight', None],
                    help="How to handle imbalanced classes for classification/discritized survival.\n"
                        "'resample' means we weighted random resampling with replacement to create an artifically balanced training set each epoch i.e. using WeightedRandomSampler() in the data loader."
                        "'loss_weight' means we weight each sample in the loss function to downweight large classes/upweight small classes\n"
                        "'none' means we do nothing to address class imbalance.")

parser.add_argument('--mini',
                    action='store_true', default=False,
                    help='Run mini experiment for debugging.')

# architecture
parser.add_argument('--arch', type=str, default='amil_slim',
                    choices=['amil_slim', 'amil_nn', 'amil_var_nn', 'sum_mil', 'sum_var_mil', 'patchGCN', 'patchGCN_varpool'],
                    help="Which neural network architecture to use.\n"
                         "'amil_slim' just does attention mean pooling with a final linear layer.\n"
                         "'amil_nn' does attention mean pooling with an additional neural network layers applied to the instance embeddings and the mean pooled output.")

parser.add_argument('--dropout',
                    action='store_true', default=False,
                    help='Use dropout (p=0.25 by default).')

parser.add_argument('--final_nonlin', default='relu', type=str,
                    choices=['relu', 'tanh', 'identity'],
                    help='Choice of final nonlinearity for architecture.')


parser.add_argument('--attn_latent_dim', default=256, type=int,
                    help='Dimension of the attention latent space.')

parser.add_argument('--mode', default='patch', type=str,
                    choices=['patch', 'graph'],
                    help='Framework paradigm. patch for independent path assumption. graph for contextual-aware algorithms')

# For everything but amil slim
parser.add_argument('--head_n_hidden_layers', default=1, type=int,
                    help='Number of hidden layers in the head network (excluding the final output layer).')


parser.add_argument('--head_hidden_dim', default=256, type=int,
                    help='Dimension of the head hidden layers.')


# For var pool
parser.add_argument('--n_var_pools', default=100, type=int,
                    help='Number of projection vectors for variance pooling.')

parser.add_argument('--var_act_func', default='sigmoid', type=str,
                    choices=['sqrt', 'log', 'sigmoid', 'identity'],
                    help='Activation function for var pooling.')

parser.add_argument('--separate_attn',
                    action='store_true', default=False,
                    help='Use separate attention branches for the mean and variance pools.')


parser.add_argument('--freeze_var_epochs',
                    default=None, type=int,
                    help='Freeze the variance pooling weights for an initial number of epochs.')

# For rank loss
parser.add_argument('--rank_loss_phi', default='sigmoid', type=str,
                    choices=['sigmoid', 'relu'],
                    help='The phi function for rank loss.')

args = parser.parse_args()

start_time = time()

print('\n\n\n==============')
print("Start training")
print('==============')
print(args)

# load task info from yaml file
feat_dir, y_fpath, train_dir, task = parse_mil_task_yaml(fpath=args.task_fpath)
if args.mode == 'graph':
    # For GCNs assume graph files are in graph subfolder of the feat directory
    feat_dir = os.path.join(feat_dir, 'graph')

########################
# Identify/Assign gpus #
########################
device = assign_free_gpus()

set_seeds(device=device, seed=args.seed)

##################################
# Setup paths for saving results #
##################################
if args.name == 'time':
    args.name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

train_dir = os.path.join(train_dir, args.name) \
    if args.name is not None else train_dir

# training logs, model checkpoints, final results file
log_dir = join_and_make(train_dir, 'log')
checkpoint_dir = join_and_make(train_dir, 'checkpoints')
checkpoint_fpath = os.path.join(checkpoint_dir, 's_checkpoint.pt')

train_preds_fpath = os.path.join(train_dir, 'train_preds')
val_preds_fpath = os.path.join(train_dir, 'val_preds')


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
    y_train = y_split['train']
    y_val = y_split['val']

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

    y_train = y_split['train']
    y_val = y_split['val']
    n_time_bins = len(set(y_train['time_bin'].values))

else:
    raise NotImplementedError("Not implemented for {}".format(task))

# samples in train/test sets
train_samples = y_train.index.values
val_samples = y_val.index.values

#####################################
# Format mini experiment parameters #
#####################################

if args.mini:
    args.n_epochs = 3

    # train_samples = train_samples[0:3]
    # val_samples = val_samples[0:3]

    # sample a subset of slides
    n_samps = 3
    if task == 'clf':

        train_samples = pd.DataFrame(y_train).\
            groupby('label', group_keys=False).\
            apply(lambda x: x.sample(n_samps)).index.values

        val_samples = pd.DataFrame(y_val).\
            groupby('label', group_keys=False).\
            apply(lambda x: x.sample(n_samps)).index.values

    elif task in ['discr_surv', 'rank_surv', 'cox_surv']:

        train_samples = pd.DataFrame(y_train).\
            groupby('time_bin', group_keys=False).\
            apply(lambda x: x.sample(n_samps)).index.values

        val_samples = pd.DataFrame(y_val).\
            groupby('time_bin', group_keys=False).\
            apply(lambda x: x.sample(n_samps)).index.values
    else:
        # Not implemented for Cox
        raise NotImplementedError("Not implemented for {}".format(task))

######################
# setup for datasets #
######################

# file paths containing features features
# TODO: uncomment when we remove TCGA specific code below
# train_fpaths = find_fpaths(folders=feat_dir, ext=['h5', 'pt'],
#                            names=train_samples)
# val_fpaths = find_fpaths(folders=feat_dir, ext=['h5', 'pt'],
#                          names=val_samples)

# format these to the patient level
# TODO: eventually remove this from var_pool -- just used for experiments in paper!
train_fpaths = find_fpaths(folders=feat_dir, ext=['h5', 'pt'])
train_fpaths = tcga_agg_slides_to_patient_level(train_fpaths,
                                                names=train_samples)

val_fpaths = find_fpaths(folders=feat_dir, ext=['h5', 'pt'])
val_fpaths = tcga_agg_slides_to_patient_level(val_fpaths,
                                              names=val_samples)


if len(train_fpaths) == 0:
    raise RuntimeError("No training files found in {}".format(feat_dir))
if len(val_fpaths) == 0:
    # TODO: maybe warn?
    raise RuntimeError("No validation files found in {}".format(feat_dir))

# Data loading bureaucracy
dataset_kws = {}
loader_kws = {'num_workers': 4} if device.type == "cuda" else {}

if args.mode == 'patch':
    dataset_train = BagDataset(bag_fpaths=train_fpaths, y=y_train, task=task,
                               mode=args.mode, **dataset_kws)
    dataset_val = BagDataset(bag_fpaths=val_fpaths, y=y_val, task=task,
                             mode=args.mode, **dataset_kws)

    n_instances_train, n_bag_feats = dataset_train.get_bag_summary()
elif args.mode == 'graph':
    ds_train = GraphDataset(feat_dir, y_train, task)
    dataset_train = ds_train()
    ds_val = GraphDataset(feat_dir, y_val, task)
    dataset_val = ds_val()

    # A bit hacky way, but this will do
    n_bag_feats = dataset_train[0].x.shape[1]

##########################
# Handle class imbalance #
##########################
sampler = None
if task == 'clf':

    if args.imbal_how == 'resample':

        sample_weights = get_weights_for_balanced_clf(dataset_train.y)

        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

    elif args.imbal_how == 'loss_weight':
        # setup class weights to address imbalanced class sizes
        # weight by inverse frequency of training counts
        # TODO: maybe make this optional?
        loss_class_weights = (1 / y_train.value_counts())
        loss_class_weights /= loss_class_weights.sum()
        loss_class_weights = torch.from_numpy(loss_class_weights.values).float()
        loss_class_weights = loss_class_weights.to(device)


elif task == 'discr_surv':

    if args.imbal_how == 'resample':

        sample_weights = \
            get_weights_for_balanced_binXcensor(surv_df=dataset_train.y,
                                                time_bin_col='time_bin',
                                                censor_col='censorship')

        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

    elif args.imbal_how == 'loss_weight':
        raise NotImplementedError("TODO: add loss weights for discr_surv")


######################
# Setup data loaders #
######################
if task == 'rank_surv' and args.batch_size < 2:
    raise ValueError("For rank_surv we need a batch size of at least 2")

if args.mode == 'patch':

    # Set fixed bag size
    if args.fixed_bag_size in [None, 'max']:
        fixed_bag_size = args.fixed_bag_size

    elif isinstance(args.fixed_bag_size, str)\
            and args.fixed_bag_size[0] == 'q':
        # fixed bag size is qth quntile
        # pull out quantile
        q = float(args.fixed_bag_size[1:])
        fixed_bag_size = int(np.percentile(n_instances_train, q=q))
    else:
        fixed_bag_size = int(args.fixed_bag_size)
    print("using fixed_bag_size", fixed_bag_size)

    # ensures all bags have same number of instances
    # possibly subsamples instances
    collate_fn = get_collate_fixed_bag_size(fixed_bag_size) \
        if fixed_bag_size is not None else None

    # check batch size/fixed bag size
    if args.batch_size > 1 and args.fixed_bag_size is None:
        raise RuntimeError("If batch_size > 1 you have to set fixed_bag_size."
                           "If you want to ensure every instance is included "
                           "in each bag use --fixed_bag_size max.")

elif args.mode == 'graph':
    pass
    # if args.batch_size > 1:
    #     raise RuntimeError("For now, use batch size of 1 for patchGCN")

else:
    raise NotImplementedError("Not implemented for mode {}".format(args.mode))


# turn shuffle of if we are using a smapler
shuffle = sampler is None

if task == 'rank_surv':
    # drop last batch to avoid accidently having a batch size of 1
    # TODO: figuore out a more elegant way of handling this issue
    # without having to drop loas
    drop_last = True
else:
    drop_last = False

# loader used for training
# BEWARE subtleties in how we compute the loss functions for rank/cox loss
# for rank loss these loaders will return random sets of comparable pairs
# so we probably don't see every comparable pair in the loss in a given
# training epoch
# for cox each event observation will get a random subset of its risk set
# so we aren't actually computing the true cox loss, but an approxmation'

if args.mode == 'patch':

    loader_train = DataLoader(dataset_train,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              sampler=sampler,
                              shuffle=shuffle,
                              drop_last=drop_last,
                              **loader_kws)

    # used to evaulate the validation metrics
    loader_val = DataLoader(dataset_val, batch_size=1,
                            shuffle=True, drop_last=False,
                            **loader_kws)

    # loader used for evaluating the training set e.g. doesn't use a sampler
    loader_train_eval = DataLoader(dataset_train, batch_size=1,
                                   shuffle=True, drop_last=False,
                                   **loader_kws)

elif args.mode == 'graph':
    # For GNN, use torch_geometric data loader
    loader_train = GraphDataLoader(dataset_train,
                                   batch_size=args.batch_size,
                                   shuffle=shuffle,
                                   **loader_kws)

    # used to evaulate the validation metrics
    loader_val = GraphDataLoader(dataset_val,
                                 batch_size=1,
                                 shuffle=True,
                                 **loader_kws)

    # loader used for evaluating the training set e.g. doesn't use a sampler
    loader_train_eval = GraphDataLoader(dataset_train,
                                        batch_size=1,
                                        shuffle=True,
                                        **loader_kws)


#######################
# Setup Loss function #
#######################

if task == 'clf':

    out_dim = n_classes

    # loss_class_weights set above -- only not None if args.imbal_how=='loss_weight'
    loss_func = nn.CrossEntropyLoss(weight=loss_class_weights)

elif task == 'discr_surv':
    out_dim = n_time_bins

    if args.imbal_how == 'loss_weight':
        raise NotImplementedError("Need to add weighted version")

    else:
        loss_func = NLLSurvLoss(alpha=0)  # weight=loss_class_weights

elif task == 'cox_surv':
    # assert args.mode != 'graph', "Cox surv for GCN not implemented yet"

    out_dim = 1
    loss_func = CoxLoss(reduction='mean')

elif task == 'rank_surv':
    # assert args.mode != 'graph', "Rank surv for GCN not implemented yet"

    out_dim = 1
    loss_func = SurvRankingLoss(phi=args.rank_loss_phi, reduction='mean')

else:
    raise NotImplementedError("Not implemented for {}".format(task))


# Loss function to use for evaulation
# by setting overwrite_loss_metric_key to be not None
# we obtain the loss function value from the metrics spit out
# by the Evaler object. This is useful when the torch loss function
# does not actually output the true loss function
if task in ['clf', 'discr_surv']:
    eval_loss_func = loss_func
    overwrite_loss_metric_key = None
elif task == 'cox_surv':
    eval_loss_func = None
    overwrite_loss_metric_key = 'cox_loss'
elif task == 'rank_surv':
    eval_loss_func = None
    overwrite_loss_metric_key = 'rank_loss'

###############
# Setup Model #
###############
model = get_model(args=args, n_bag_feats=n_bag_feats, out_dim=out_dim)

if args.freeze_var_epochs is not None and hasattr(model, 'var_pool_off'):
    model.var_pool_off()

print(get_network_summary(model))

initialize_weights(model)

# Initialize variace projects with unit norm scaling + isotropic gaussian dists
if hasattr(model, 'var_pool'):
    model.var_pool.init_var_projections()

# if hasattr(model, 'var_neck'):
#     # initialize var neck close to zero
#     model.var_neck.weight.data *= 1e-4

model.to(device)

# TODO: uncomment
# if torch.cuda.device_count() >= 1:
#     # if we have multiple GPUs then we can parallelize the encoder
#     # and instance attention scores. Note for this to parallelize
#     # across batches and bag instances model.enc_and_attend has to be
#     # setup carefullly (e.g. see var_pool.nn.arch.AttnMIL.AttnMeanPoolMIL)
#     model.enc_and_attend = nn.DataParallel(model.enc_and_attend)

#     # TODO: i'm not sure we are actually getting a speed up with multiple GPUs
#     # dig into this!


##################
# Setup training #
##################

if task.endswith('surv'):
    train_events = ~y_train['censorship'].values.astype(bool)
    train_times = y_train['survival_time'].values
    surv_evaler_kws = {'train_events': train_events,
                       'train_times': train_times}


# setup evaluator
if task == 'clf':
    evaler = ClfEvaler(class_names=class_names)

elif task == 'discr_surv':
    evaler = DiscreteSurvivalEvaler(**surv_evaler_kws)

elif task == 'cox_surv':
    evaler = CoxSurvivalEvaler(**surv_evaler_kws)

elif task == 'rank_surv':
    evaler = RankSurvivalEvaler(phi=args.rank_loss_phi, **surv_evaler_kws)

else:
    raise NotImplementedError("Not implemented for {}".format(task))


# Which score early stopping should monitor
if args.es_monitor == 'metric':
    if task == 'clf':
        early_stop_metric = 'auc'
        min_good = False  # for early stopper

    elif task == 'discr_surv':
        early_stop_metric = 'c_index'
        min_good = False

    elif task == 'cox_surv':
        early_stop_metric = 'c_index'
        min_good = False

    elif task == 'rank_surv':
        early_stop_metric = 'c_index'
        min_good = False

    else:
        raise NotImplementedError("Not implemented for {}".format(task))

elif args.es_monitor == 'loss':
    min_good = True


# setup early stopping
if args.early_stopping:
    early_stopper = EarlyStopper(save_dir=checkpoint_dir,
                                 min_epoch=10,
                                 patience=args.stop_patience,
                                 patience_min_improve=0.001,
                                 abs_scale=True,
                                 min_good=min_good,  # be careful to set this!
                                 verbose=True
                                 )
else:
    early_stopper = None

##########################
# print data set summary #
##########################

n_tot = len(train_samples) + len(val_samples)
print("\n\nTraining {} task with {} total slides.".format(task, n_tot))
print("{} ({:1.2f}%) training slides, {} ({:1.2f}%) validation slides".
      format(len(train_samples), len(train_samples) / n_tot,
             len(val_samples), len(val_samples) / n_tot
             ))

if task == 'clf':
    print("{} classes".format(n_classes))
    counts, props = get_counts_and_props(y=y_train.values,
                                         class_names=class_names)
    print(counts)
    print(props)
elif task == 'discr_surv' or task == 'cox_surv' or task == 'rank_surv':
    print("censorship train = {:1.1f}%, val = {:1.1f}%".
          format(100 * y_train['censorship'].mean(),
                 100 * y_val['censorship'].mean()))
    print("{} time bins".format(n_time_bins))
else:
    raise NotImplementedError("Not implemented for {}".format(task))

print("Number of bag features {}".format(n_bag_feats))
print("Training bag size descrptive statistics")

if args.mode == 'patch':
    train_bag_size_summary = descr_stats(n_instances_train)
    print(train_bag_size_summary)
elif args.mode == 'graph':
    train_bag_size_summary = None


##################################
# Compute metrics for best model #
##################################
model.load_state_dict(torch.load(checkpoint_fpath),
                      map_location=device)


final_train_loss, final_train_metrics = \
    eval_loop(model=model,
              loss_func=eval_loss_func,
              loader=loader_train_eval,  # no sampler!
              evaler=evaler,
              device=device,
              tqdm_desc='Final evaluation, training set',
              mode=args.mode)

# for cox/rank loss the train_loop does not compute
# the loss properly so we obtain it from the Evaler object
if overwrite_loss_metric_key is not None:
    final_train_loss = final_train_metrics[overwrite_loss_metric_key]


# save train predictions to disk
evaler.save_tracked_data(train_preds_fpath, sample_ids=dataset_train.bag_names)

final_val_loss, final_val_metrics = \
    eval_loop(model=model,
              loss_func=eval_loss_func,
              loader=loader_val,
              evaler=evaler,
              device=device,
              tqdm_desc='Final evaluation, validation set',
              mode=args.mode)

# for cox/rank loss the train_loop does not compute
# the loss properly so we obtain it from the Evaler object
if overwrite_loss_metric_key is not None:
    final_val_loss = final_val_metrics[overwrite_loss_metric_key]

# save val predictions to disk
evaler.save_tracked_data(val_preds_fpath, sample_ids=dataset_val.bag_names)
