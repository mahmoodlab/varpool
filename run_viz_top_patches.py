import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from var_pool.utils import format_command_line_args
from var_pool.file_utils import join_and_make
from var_pool.viz.utils import save_fig


parser = argparse.\
    ArgumentParser(description='Runs a single fit experiment in parallel over a set of subtypes and dataset seeds.')

parser.add_argument('--feats_top_dir',
                    type=str, help='Where the features are saved.')

parser.add_argument('--output_dir',
                    type=str, help='Where the output should be saved.')
args = parser.parse_args()

device = None

endpoint = 'pfi'
task = 'rank_surv'

base_arch = 'amil_nn'
var_arch = 'amil_var_nn'

n_patients = 10

###############
# Setup paths #
###############

for subtype, dataset in zip(['brca', 'blca', 'coadread', 'gbmlgg', 'ucec'],
                            [10, 20, 10, 10, 80]):
    # these are the seeds with the best validation errors
    # lets use these for visualization

    wsi_dir = os.path.join(args.top_data_dir, 'wsi/tcga', subtype)

    feat_h5_dir = os.path.join(args.top_data_dir, 'mil-h5_files', subtype,
                               'resnet50_trunc_h5_patch_features')

    results_dir = os.path.join(args.output_dir, 'surv_train_out',
                               '{}-{}'.format(subtype, endpoint),
                               'dataset_{}'.format(dataset),
                               task)

    autogen_fpath = os.path.join(args.output_dir, 'autogen',
                                 'process_list_autogen-{}.csv'.format(subtype))

    # Paths for saving
    top_save_dir = join_and_make(args.output_dir, 'viz',
                                 '{}-{}-ds_{}'.format(subtype, endpoint, dataset))

    ##############################################
    # Make visualizations for base and var archs #
    ##############################################

    for i in range(2):
        if i == 0:
            arch = var_arch
        else:
            arch = base_arch

        #######################
        # Paths for this arch #
        #######################

        save_dir = join_and_make(top_save_dir, arch)

        checkpoint_fpath = os.path.join(results_dir, arch, 'checkpoints',
                                        's_checkpoint.pt')

        train_preds_fpath = os.path.join(results_dir, arch, 'train_preds.npz')
        val_preds_fpath = os.path.join(results_dir, arch, 'val_preds.npz')

        y_fpath = os.path.join(save_dir, 'response.csv')
        y_fig_fpath = os.path.join(save_dir, 'risk_preds.png')

        ####################################
        # Get highest/lowest risk patients #
        ####################################

        train_preds = np.load(train_preds_fpath)
        val_preds = np.load(val_preds_fpath)
        z = np.concatenate([train_preds['z'], val_preds['z']])
        y_true = np.vstack([train_preds['y_true'], val_preds['y_true']])
        sample_ids = np.concatenate([train_preds['sample_ids'],
                                     val_preds['sample_ids']])

        pred_risk = pd.Series(z, index=sample_ids, name='pred_risk')
        pred_risk = pred_risk.sort_values(ascending=False)  # highest risk first

        # use the var arch's predictions for determine
        # the higest/lowest risk patients
        if i == 0:
            high_risk = pred_risk.index.values[0:n_patients]
            low_risk = pred_risk.index.values[-n_patients:]

        #############################
        # Save survival predictions #
        #############################

        y_df = pd.DataFrame(y_true, index=sample_ids,
                            columns=['censor', 'survival_time'])
        y_df['censor'] = y_df['censor'].astype(bool)
        y_df = y_df.loc[pred_risk.index]
        y_df['pred_risk'] = pred_risk
        y_df.to_csv(y_fpath)

        # Plot predictions
        plt.figure(figsize=(8, 8))
        plt.scatter(y_df.query("censor")['pred_risk'],
                    y_df.query("censor")['survival_time'],
                    marker='o',
                    color='lightcoral',
                    label='censored')

        plt.scatter(y_df.query("not censor")['pred_risk'],
                    y_df.query("not censor")['survival_time'],
                    marker='x',
                    color='red',
                    label='observed')
        plt.legend()
        plt.xlabel("Predicted risk")
        plt.ylabel("Survival time")
        save_fig(y_fig_fpath)

        ##########################
        # Format command and run #
        ##########################

        command_args = {'autogen_fpath': autogen_fpath,
                        'checkpoint_fpath': checkpoint_fpath,
                        'wsi_dir': wsi_dir,
                        'feat_h5_dir': feat_h5_dir,
                        'high_risk': high_risk,
                        'low_risk': low_risk,
                        'save_dir': save_dir}

        model_args = {'arch': arch,
                      'n_var_pools': 10,
                      'var_act_func': 'log',
                      }

        model_flags = ['dropout']

        command = 'python tcga_scripts/viz_top_patches.py'
        arg_str = format_command_line_args(kws={**command_args, **model_args},
                                           flags=model_flags)

        command += ' ' + arg_str
        # print(command)

        if device is not None:
            command = 'CUDA_VISIBLE_DEVICES={} '.format(device) + command

        os.system(command)
