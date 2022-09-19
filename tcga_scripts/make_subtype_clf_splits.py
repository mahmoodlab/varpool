"""
Preprocessing for classification task between two or more cancer subtypes;  make train/val/test splits and save response data frame.
"""
import os
from glob import glob
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

from var_pool.file_utils import get_file_names

parser = argparse.\
    ArgumentParser(description='Makes the response data frame for a toy cancer subtype classification task.')

parser.add_argument('--folders', nargs='+', required=True,
                    help='Folders containing the subtype features, one folder per subtype.')

parser.add_argument('--subtypes', nargs='+',  required=True,
                    help='Names of the subtypes.')

parser.add_argument('--save_dir', type=str,  required=True,
                    help='Where the response data frame should be stored.')

args = parser.parse_args()

assert len(args.folders) == len(args.subtypes), \
    "Make sure to provide the same number of names as folders!"\
    "{} folder arguments and {} subtypes arguments provided".\
    format(len(args.folders), len(args.subtypes))

os.makedirs(args.save_dir, exist_ok=True)

exts = ['pt', 'h5']

# make subtype response for available files
y = []
for subtype, folder in zip(args.subtypes, args.folders):

    # pull out all file paths in the subtype folder matching this extension
    fpaths = []
    for ext in exts:
        fpaths_this_ext = glob(os.path.join(folder, '*.{}'.format(ext)))
        fpaths.extend(fpaths_this_ext)
    fnames = get_file_names(fpaths)

    print("{} files found for {}".format(len(fpaths), subtype))
    y.append(pd.Series(subtype, index=fnames, name='label'))

y = pd.concat(y)
y.index.name = 'sample_id'
y = pd.DataFrame(y)

# make train test splits
train_idxs, val_idxs = train_test_split(y.index,
                                        train_size=.8,
                                        shuffle=True,
                                        random_state=1,
                                        stratify=y)

y['split'] = None
y.loc[train_idxs, 'split'] = 'train'
y.loc[val_idxs, 'split'] = 'val'


# Save to disk
os.makedirs(args.save_dir, exist_ok=True)
fpath = os.path.join(args.save_dir, 'clf_{}.csv'.
                     format('_'.join(args.subtypes)))

y.to_csv(fpath)
