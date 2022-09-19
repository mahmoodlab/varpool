import os
import argparse

import torch

from var_pool.script_utils import parse_mil_task_yaml
from var_pool.mhist.get_model_from_args import get_model
from var_pool.gpu_utils import assign_free_gpus
from var_pool.file_utils import join_and_make
from var_pool.nn.datasets.VisualDatasets import VisualDataset

from var_pool.visualization.vis_utils import get_top_patches

parser = argparse.\
    ArgumentParser(description='Creates several visulaizations')

parser.add_argument('--task_fpath', type=str,
                    # default to task.yaml file saved in current directory
                    default='yaml/luad_vs_lusc.yaml',
                    help='Filepath to .yaml file containing the information '
                         'for  this task. It should include entries for: \n'
                         'feats_dir =  directory containing WSI bag features as .h5 files.\n'
                         'y_fpath: csv file containing the response labels for each bag and the train/val/test spilts. See code for how the csv file should be formatted.\n'
                         'task: a string indicated which kind of task we are solving. Should be one of "clf", "surv_cov" or "surv_discr"\n'
                         'train_dir: directory to where training results are saved e.g. model checkpoints and logging information.')

parser.add_argument('--save_dir', type=str, default=None,
                    help='(Optional) Directory where to save the results. If not provided, will use the original train directory from the yaml file.')

parser.add_argument('--name', type=str, default=None,
                    help='(Optional) Name of this experiment; the results will be saved in train_dir/name/. If name=time, will name the directory after the current date ane time.')

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


# For var pool
parser.add_argument('--n_var_pools', default=10, type=int,
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

# For visualization
parser.add_argument('--n_proj', default=5, type=int,
                    help='Number of projections to visualize')

parser.add_argument('--n_patches', default=5, type=int,
                    help='Number of patches to visualize')

parser.add_argument('--slide_num', default=0, type=int,
                    help='Slide number')

parser.add_argument('--image_dir', type=str, default=None,
                    help="Where the svs images are saved")

args = parser.parse_args()

# load task info from yaml file
feat_dir, y_fpath, train_dir, task = parse_mil_task_yaml(fpath=args.task_fpath)

########################
# Identify/Assign gpus #
########################
device = assign_free_gpus()

# Create folders/paths
# where to load the trained results
train_dir = os.path.join(train_dir, args.name) \
    if args.name is not None else train_dir
checkpoint_dir = join_and_make(train_dir, 'checkpoints')
checkpoint_fpath = os.path.join(checkpoint_dir, 's_checkpoint.pt')

save_dir = args.save_dir if args.save_dir is not None else train_dir
# Directory to images
assert args.image_dir is not None, "You need to supply .svs images folder"
image_dir = args.image_dir

#######################
# Assume single image #
#######################
dataset_kws = {}
loader_kws = {'num_workers': 4} if device.type == "cuda" else {}

dataset = VisualDataset(feat_dir, image_dir)
n_bag_feats = dataset.get_feat_dim()
slide_coords, slide_feats = dataset[args.slide_num]
slide_feats = slide_feats.unsqueeze(0)  # (1 x n_instances x feat_dim)

print("Loading the model...")
if task == 'discr_surv':
    out_dim = 4
elif task == 'cox_surv':
    out_dim = 1
elif task == 'rank_surv':
    out_dim = 1
model = get_model(args=args, n_bag_feats=n_bag_feats, out_dim=out_dim)
state_dict = torch.load(checkpoint_fpath)

# Hack
# state_dict['head.1.bias'] = state_dict.pop('head.2.bias')
# state_dict['head.1.weight'] = state_dict.pop('head.2.weight')

# Load checkpoints
model.load_state_dict(state_dict)
model.to(device)

print("Forward pass through the model...")
with torch.no_grad():
    slide_feats = slide_feats.to(device)
    z = model(slide_feats)

print("Query the patches...")
# Obtain scores & projection vectors
if hasattr(model, 'var_pool'):
    var_pooled_feats = model.get_variance(slide_feats, normalize=False)
    # Identify top proj vectors by variance for now
    varpool_val, varpool_idx = torch.topk(var_pooled_feats,
                                          args.n_proj,
                                          largest=True)

    highest_list = []
    lowest_list = []
    # For each projection direction, get highest-scoring patches
    for idx, val in list(zip(varpool_idx.squeeze(), varpool_val.squeeze())):
        var_proj_vec = model.var_pool.get_projection_vector(idx)
        highest_patches = get_top_patches(model.encode(slide_feats),
                                          var_proj_vec,
                                          n_patches=args.n_patches)

        lowest_patches = get_top_patches(model.encode(slide_feats),
                                         var_proj_vec,
                                         n_patches=args.n_patches,
                                         largest=False)

        highest_list.append(highest_patches)
        lowest_list.append(lowest_patches)

##############
# Load image #
##############
print("Reading images...")
highest_img_list = []
for idx, _, _ in highest_list:
    img = dataset.read_patch(args.slide_num, slide_coords[idx])
    highest_img_list.append(img)

lowest_img_list = []
for idx, _, _ in lowest_list:
    img = dataset.read_patch(args.slide_num, slide_coords[idx])
    lowest_img_list.append(img)
