import argparse
import os
from multiprocessing import Pool
import torch
from itertools import product
import yaml
import pandas as pd

parser = argparse.\
    ArgumentParser(description='Runs a single fit experiment in parallel over a set of subtypes and dataset seeds.')

parser.add_argument('--feats_top_dir',
                    type=str, help='Where the features are saved.')

parser.add_argument('--output_dir',
                    type=str, help='Where the output should be saved.')


parser.add_argument('--task', type=str, help='Which loss task.',
                    choices=['rank_surv', 'cox_surv', 'discr_surv'])

parser.add_argument('--arch_kind', type=str,
                    choices=['amil', 'deepsets', 'amil_gcn',
                             'patch_gcn', 'all'],
                    help='Which architecture kind.')

parser.add_argument('--cuda', default=0, type=int,
                    choices=[0, 1, 2],
                    help='To manually parallelize the process')

args = parser.parse_args()

endpoint = 'pfi'
subtypes = ['blca', 'brca', 'coadread', 'gbmlgg', 'ucec']
dataset_seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


################################
# Setup high level directories #
################################

n_devices = torch.cuda.device_count()

if args.arch_kind in ['amil_gcn', 'patch_gcn']:
    if args.task == 'discr_surv':
        train_params = '--seed 1 --dropout '\
            '--n_epochs 30 --lr 2e-4 --batch_size 1 --fixed_bag_size q75 '\
            '--n_var_pools 10 --var_act_func log --mode graph '\
            '--grad_accum 32 --imbal_how resample'
    else:
        train_params = '--seed 1 --dropout '\
            '--n_epochs 30 --lr 1e-4 --batch_size 32 --fixed_bag_size q75 '\
            '--n_var_pools 10 --var_act_func log --mode graph '\
            '--grad_accum 1 --imbal_how resample'
else:
    train_params = '--seed 1 --dropout '\
        '--n_epochs 30 --lr 2e-4 --batch_size 32 --fixed_bag_size q75 '\
        '--n_var_pools 10 --var_act_func log --imbal_how resample '\
        '--grad_accum 1'

if args.arch_kind == 'amil':
    archs2run = ['amil_nn', 'amil_var_nn']
elif args.arch_kind == 'deepsets':
    archs2run = ['sum_mil', 'sum_var_mil']
elif args.arch_kind == 'amil_gcn':
    archs2run = ['amil_gcn_varpool', 'amil_gcn']
elif args.arch_kind == 'patch_gcn':
    archs2run = ['patchGCN', 'patchGCN_varpool']
elif args.arch_kind == 'all':
    archs2run = ['amil_nn', 'amil_var_nn', 'sum_mil', 'sum_var_mil']


####################################
# Create setup for each experiment #
####################################
train_commands = []
for subtype in subtypes:
    for dataset in dataset_seeds:

        ########################
        # Make survival splits #
        ########################

        tcga_clincal_fpath = os.path.join(args.output_dir,
                                          'clinical_data',
                                          'TCGA-CDR-union-gbmlgg-coadread.xlsx')

        surv_respose_dir = os.path.join(args.output_dir, 'surv_response',
                                        '{}-{}'.format(subtype,
                                                       endpoint),
                                        'dataset_{}'.format(dataset))

        if args.arch_kind in ['amil_gcn', 'patch_gcn']:
            feats_dir = os.path.join(args.feats_top_dir, subtype, 'graph')
        else:
            feats_dir = os.path.join(args.feats_top_dir, subtype)

        make_splits_kws = {'feats_dir': feats_dir,
                           'tcga_clincal_fpath': tcga_clincal_fpath,
                           'surv_respose_dir': surv_respose_dir,
                           'subtype': subtype,
                           'endpoint': endpoint,
                           'dataset': dataset
                           }

        make_splits_command = 'python tcga_scripts/make_discr_suvr_splits.py '\
            '--tcga_clincal_fpath {tcga_clincal_fpath} '\
            '--feats_dir {feats_dir} --save_dir {surv_respose_dir} '\
            '--subtype {subtype} --endpoint {endpoint} '\
            '--prop_trian .7 --seed {dataset} --n_bins 4 --no_test_split'.\
            format(**make_splits_kws)

        os.system(make_splits_command)

        #######################
        # Make task yaml file #
        #######################

        y_fpath = os.path.join(surv_respose_dir, 'discr_survival.csv')

        task_fpath = os.path.join(output_dir, 'surv_yaml',
                                  '{}-{}-ds_{}-{}.yaml'.
                                  format(subtype, endpoint,
                                         dataset, args.task),
                                  )

        train_dir = os.path.join(output_dir, 'surv_train_out',
                                 '{}-{}'.format(subtype, endpoint),
                                 'dataset_{}'.format(dataset),
                                 args.task)

        make_yaml_kws = {'feats_dir': feats_dir,
                         'train_dir': train_dir,
                         'y_fpath': y_fpath,
                         'task_fpath': task_fpath,
                         'task': args.task
                         }

        make_yaml_command = 'python tcga_scripts/make_surv_yaml.py '\
            '--fpath {task_fpath} --task {task} --y_fpath {y_fpath} '\
            '--feats_dir {feats_dir} --train_dir {train_dir}'.\
            format(**make_yaml_kws)

        os.system(make_yaml_command)

        ############################
        # Compute stat sig c-index #
        ############################
        stat_sig_dir = os.path.join(output_dir, 'c_index_stat_sig')

        stat_sig_kws = {'response_fpath': y_fpath,
                        'stat_sig_dir': stat_sig_dir,
                        'save_stub': '{}-{}-dataset_{}'.
                        format(subtype, endpoint, dataset)}

        stat_sig_command = 'python tcga_scripts/stat_sig_c_index_cutoff.py '\
            '--response_fpath {response_fpath} '\
            '--save_dir {stat_sig_dir} --save_stub {save_stub}'.\
            format(**stat_sig_kws)

        os.system(stat_sig_command)

        #############################
        # Make run train.py command #
        #############################
        for arch in archs2run:

            run_train_kws = {'task_fpath': task_fpath,
                             'cuda': args.cuda,
                             'name': arch,
                             'arch': arch,
                             'train_params': train_params}

            run_train_command = 'CUDA_VISIBLE_DEVICES={cuda} '\
                                'python scripts/train.py '\
                                '--task_fpath {task_fpath} --name {name} '\
                                '--arch {arch} {train_params}'.\
                                format(**run_train_kws)

            train_commands.append(run_train_command)

# pool = Pool(processes=n_devices)
pool = Pool(processes=1)    # For manual GPU allocation
pool.starmap(os.system, list(zip(train_commands)))
pool.close()
pool.join()

#############################
# Aggregate results for val #
#############################
save_dir = os.path.join(args.output_dir, 'fit_results')
os.makedirs(save_dir, exist_ok=True)
missing = []
for subtype in subtypes:

    results = []
    for dataset, arch in product(dataset_seeds, archs2run):

        # load results for one experiment
        res_fpath = os.path.join(args.output_dir, 'surv_train_out',
                                 '{}-{}'.format(subtype, endpoint),
                                 'dataset_{}'.format(dataset),
                                 args.task, arch,
                                 'results.yaml')

        if os.path.exists(res_fpath):
            with open(res_fpath) as file:
                res = yaml.safe_load(file)
            res['dataset'] = dataset
            res['arch'] = arch
            results.append(res)

        else:
            missing.append({'subtype': subtype, 'dataset': dataset,
                            'arch': arch})

    # Save results for this subtype
    results_fpath = os.path.join(save_dir,
                                 'results-{}-{}-{}_{}_val.csv'.
                                 format(subtype, endpoint, args.task,
                                        args.arch_kind))
    results = pd.DataFrame(results)
    results.to_csv(results_fpath, index=False)

missing_fpath = os.path.join(save_dir, 'missing_results_val.csv')
missing = pd.DataFrame(missing)
missing.to_csv(missing_fpath, index=False)
