import os
import yaml
import pandas as pd
from glob import glob
import argparse


raise NotImplementedError("Need to rewrite for new folder naming strategies.")

parser = argparse.\
    ArgumentParser(description='Aggregate CV folds to get summary statistics/')

parser.add_argument('--folder', type=str, help="Folder containing the fold results as results.")

args = parser.parse_args()

# Read in results for each fold
fold_folders = glob(os.path.join(args.tune_dir, 'fold_*/'))

results = []
for folder_path in fold_folders:

    # path/fold_IDX/ -> fold_IDX
    folder_name = folder_path.split('/')[-2]
    fold_idx = int(folder_name.split('_')[1])

    # load results fpath
    results_fpath = os.path.join(folder_path, 'results.yaml')
    with open(results_fpath, 'r') as file:
        res = yaml.safe_load(file)
    res['fold'] = fold_idx

    results.append(res)

# format results to pandas
results = pd.DataFrame(results)
results = results.set_index('fold')
results = results[['val_c_index', 'val_loss',
                   'train_c_index', 'train_loss',
                   'runtime']]  # reorder columns

# compute fold summary statistis
results_summary = results.agg(['mean', 'std', 'min', 'max', 'median']).T

# write output string
output = ''
output += '{} cross-validation results\n'.format(args.subtype)
output += 'Validation c-index avg = {:1.2f}, std={:1.2f}'.\
          format(100 * results_summary.loc['val_c_index', 'mean'],
                 100 * results_summary.loc['val_c_index', 'std'])
output += '\n\n\n'

output += 'Individual fold results'
output += str(results)
output += '\n\nFold summary statistics'
output += str(results_summary)

print(output)

save_fpath = os.path.join(args.folder, 'results_agg.txt')
with open(save_fpath, 'w') as f:
    f.write(output)
