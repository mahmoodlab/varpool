import os
import pandas as pd
import argparse

from var_pool.processing.discr_surv_utils import get_perm_c_index_quantile

parser = argparse.\
    ArgumentParser(description='Computes the cutoff for statistically significant c-index for a given task.')

parser.add_argument('--response_fpath', type=str,
                    help="Path to response file.")

parser.add_argument('--subtype', type=str,
                    choices=['brca', 'gbmlgg', 'blca', 'ucec', 'luad'],
                    help="Which cancer subtype.")

parser.add_argument('--endpoint', type=str,
                    choices=['os', 'pfi', 'dfi', 'dss'],
                    help="Which survival endpoint.")

parser.add_argument('--save_dir', type=str, default=None,
                    help="Folder where to save the cutoff data."
                         "If not provided, will just print out results.")

parser.add_argument('--save_stub', type=str, default=None,
                    help="Name stub for the saved file.")

args = parser.parse_args()

if args.save_dir is not None:
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_stub is not None:
        fname = '{}-c_index_stat_sig.csv'.format(args.save_stub)
    else:
        fname = 'c_index_stat_sig.csv'
    save_fpath = os.path.join(args.save_dir, fname)

df = pd.read_csv(args.response_fpath)
results = []
for kind in ['train', 'val', 'test']:
    kind_df = df.query("split=='{}'".format(kind))

    # if no samples in this split then dont bother
    if kind_df.shape[0] == 0:
        continue

    event = ~kind_df['censorship'].values.astype(bool)
    time = kind_df['survival_time'].values

    ci = get_perm_c_index_quantile(event=event, time=time,
                                   n_perm=1000, q=[.5, .95, .99])

    results.append({'split': kind,
                    'null_ci_95': ci[1],
                    'null_ci_99': ci[2]})

results = pd.DataFrame(results)

print("C-index statistical significance cutoffs for {}".\
      format(args.response_fpath))
print(results)

if args.save_dir is not None:
    results.to_csv(save_fpath)
