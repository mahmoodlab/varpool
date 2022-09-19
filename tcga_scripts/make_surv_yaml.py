import argparse
import yaml
import os

parser = argparse.\
    ArgumentParser(description='Makes a yaml file for a survival prediction task.')


parser.add_argument('--fpath', type=str, required=True,
                    help='File path for the name of this yaml file.')


parser.add_argument('--feats_dir', type=str, required=True,
                    help='Directory where the features are stored')


parser.add_argument('--y_fpath', type=str, required=True,
                    help='File path to response data frame.')

parser.add_argument('--train_dir', type=str, required=True,
                    help='Directory where the training results are stored.')

parser.add_argument('--task', type=str, required=True,
                    help="Task")


args = parser.parse_args()

data = {'feats_dir': args.feats_dir,
        'y_fpath': args.y_fpath,
        'task': args.task,
        'train_dir': args.train_dir
        }


folder = os.path.dirname(args.fpath)
os.makedirs(folder, exist_ok=True)

with open(args.fpath, 'w') as file:
    yaml.dump(data, file)
