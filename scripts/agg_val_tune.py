from glob import glob
import os
import yaml
import argparse
import pandas as pd


parser = argparse.\
    ArgumentParser(description='Aggregates validation set tuning results.')

parser.add_argument('--tune_dir', type=str,
                    help='Directory containing tune folders named tune_1, tune_2, ...')

parser.add_argument('--metric', type=str,
                    help='Name of the metric to use to pick the best model.')

parser.add_argument('--small_good',
                    action='store_true', default=False,
                    help='By default we want to maximize the tuning metric; this flag says we want to minimize the metric.')

parser.add_argument('--save_dir', type=str,
                    help='Directory where to save the aggregated tuning results.')

args = parser.parse_args()

# setup fpaths
os.makedirs(args.save_dir, exist_ok=True)
tune_results_fpath = os.path.join(args.save_dir, 'tune_results.csv')
missing_results_fpath = os.path.join(args.save_dir, 'missing_results.csv')

summary_fpath = os.path.join(args.save_dir, 'tune_summary.txt')

#######################
# Load tuning results #
#######################

# get all folders named tune_IDX
tune_folders = glob(os.path.join(args.tune_dir, 'tune_*/'))

if len(tune_folders) == 0:
    raise RuntimeError("No tuning folders found")
else:
    print("Found {} tuning folders".format(len(tune_folders)))

tune_results = []
tune_params = {}
missing_results = []
for folder_path in tune_folders:
    # path/tune_IDX/ -> tune_IDX
    folder_name = folder_path.split('/')[-2]
    tune_idx = int(folder_name.split('_')[1])

    # load tune param configs
    tune_params_fpath = os.path.join(folder_path, 'tune_params.yaml')
    with open(tune_params_fpath, 'r') as file:
        this_params = yaml.safe_load(file)
    this_params['tune_idx'] = tune_idx  # add tune index

    # try loading the results
    results_fpath = os.path.join(folder_path, 'results.yaml')
    if os.path.exists(results_fpath):

        # load results and tune params
        with open(results_fpath, 'r') as file:
            this_res = yaml.safe_load(file)

        this_res.update(this_params)  # add tuning params

        tune_results.append(this_res)
        tune_params[tune_idx] = this_params

    else:
        # missing results
        missing_results.append(this_params)


# format to pandas and save to dis
tune_results = pd.DataFrame(tune_results).\
    set_index('tune_idx').sort_index()
tune_results.to_csv(tune_results_fpath)

if len(missing_results) > 0:
    missing_results = pd.DataFrame(missing_results).\
        set_index('tune_idx').sort_index()
    missing_results.to_csv(missing_results_fpath)

####################
# Pick best metric #
####################

# pull out the metric we want to use to select the tuning parameter
scores = tune_results[args.metric].copy()
if args.small_good:
    best_tune_idx = scores.idxmin()
else:
    best_tune_idx = scores.idxmax()

best_tune_params = tune_params[best_tune_idx]
best_score = scores.loc[best_tune_idx]

scores_agg = scores.agg(['mean', 'std', 'min', 'max', 'median'])


# sort
scores.sort_values(ascending=False, inplace=True)
tune_results.sort_values(by=args.metric, ascending=False, inplace=True)

######################
# print/write output #
######################

output = ''
output += "Best tune {} score: {}\n".format(args.metric, best_score)
output += "Best tune idx: {}\n".format(best_tune_idx)
output += "Best tune params\n"
output += "{}\n\n".format(best_tune_params)
output += '{} agg\n'.format(args.metric)
output += '{}\n\n'.format(scores_agg)
output += 'all {} values\n'.format(args.metric)
output += '{}\n\n'.format(scores)
output += '{}\n\n'.format(tune_results.to_string(float_format='%1.6f'))
output += 'Missing: \n{}'.format(missing_results)

print(output)
with open(summary_fpath, 'w') as f:
    f.write(output)
