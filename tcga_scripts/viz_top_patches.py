import argparse
from itertools import chain
import os
import numpy as np  #  this fixes weird import error with torch
import torch
from tqdm import tqdm

from var_pool.mhist.get_model_from_args import get_model
from var_pool.mhist.tcga_agg_slides_to_patient_level import \
    tcga_agg_slides_to_patient_level
from var_pool.file_utils import find_fpaths, get_file_names
from var_pool.viz.top_attn import viz_top_attn_patches
from var_pool.viz.var_pool_extremes import viz_var_proj_patches_quantiles
from var_pool.nn.arch.VarPool import AttnMeanAndVarPoolMIL

parser = argparse.\
    ArgumentParser(description='Visualizes the top attended patches')

parser.add_argument('--autogen_fpath', type=str,
                    help='Path to autgen file.')

parser.add_argument('--checkpoint_fpath', type=str,
                    help='Path to model checkpoint.')

parser.add_argument('--wsi_dir', type=str,
                    help='Directory containing WSIs.')

parser.add_argument('--feat_h5_dir', type=str,
                    help='Directory containing feature hdf5 files.')

parser.add_argument('--high_risk', type=str, nargs='+',
                    help='List of high risk patients.')

parser.add_argument('--low_risk', type=str, nargs='+',
                    help='List of low risk patients.')

parser.add_argument('--save_dir', type=str,
                    help='Where to save the images.')


#################################
# Specify architecture of model #
#################################
parser.add_argument('--arch', type=str, default='amil_slim',
                    choices=['amil_slim', 'amil_nn', 'amil_var_nn', 'sum_mil', 'sum_var_mil', 'patchGCN', 'patchGCN_varpool'],
                    help="Which neural network architecture to use.\n"
                         "'amil_slim' just does attention mean pooling with a final linear layer.\n"
                         "'amil_nn' does attention mean pooling with an additional neural network layers applied to the instance embeddings and the mean pooled output.")

parser.add_argument('--final_nonlin', default='relu', type=str,
                    choices=['relu', 'tanh', 'identity'],
                    help='Choice of final nonlinearity for architecture.')

parser.add_argument('--attn_latent_dim', default=256, type=int,
                    help='Dimension of the attention latent space.')

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

args = parser.parse_args()

print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

n_top_patches = 10

##############
# Load stuff #
##############

n_bag_feats = 1024
out_dim = 1  # TODO: allow to specify for different tasks


# Load model
model = get_model(args=args, n_bag_feats=n_bag_feats, out_dim=out_dim)


state_dict = torch.load(args.checkpoint_fpath,
                        map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


# aggregate feature fpaths per patient
feat_fpaths = find_fpaths(folders=args.feat_h5_dir, ext=['h5'])
feat_fpaths = tcga_agg_slides_to_patient_level(feat_fpaths)
assert len(feat_fpaths) > 0, "No feature files found"

patient_iter = chain(zip(['low_risk'] * len(args.low_risk), args.low_risk),
                     zip(['high_risk'] * len(args.high_risk), args.high_risk)
                     )

for risk, patient_id in tqdm(list(patient_iter)):

    # get slide names for this patient
    h5_fpath = feat_fpaths[patient_id]
    slide_names = get_file_names(h5_fpath)
    wsi_fpath = [os.path.join(args.wsi_dir, name + '.svs')
                 for name in slide_names]

    #########################
    # Top atteneded patches #
    #########################

    save_fpath = os.path.join(args.save_dir, 'top_attn', risk,
                              patient_id + '.png')

    # TODO: uncomment
    # viz_top_attn_patches(model=model,
    #                      wsi_fpath=wsi_fpath,
    #                      h5_fpath=h5_fpath,
    #                      autogen_fpath=args.autogen_fpath,
    #                      device=device,
    #                      n_top_patches=n_top_patches,
    #                      save_fpath=save_fpath)

    ########################
    # Variance projections #
    ########################

    if not isinstance(model, AttnMeanAndVarPoolMIL):
        continue

    # quantiles = [0, 25, 50, 75, 100]
    quantiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    save_dir = os.path.join(args.save_dir, 'var_proj_extremes', risk)

    viz_var_proj_patches_quantiles(model=model,
                                   wsi_fpath=wsi_fpath,
                                   h5_fpath=h5_fpath,
                                   autogen_fpath=args.autogen_fpath,
                                   device=device,
                                   with_attn=True,
                                   quantiles=quantiles,
                                   n_to_show=n_top_patches,
                                   save_dir=save_dir,
                                   name=patient_id)

    save_dir = os.path.join(args.save_dir, 'var_proj_extremes-no_attn', risk)

    viz_var_proj_patches_quantiles(model=model,
                                   wsi_fpath=wsi_fpath,
                                   h5_fpath=h5_fpath,
                                   autogen_fpath=args.autogen_fpath,
                                   device=device,
                                   with_attn=False,
                                   quantiles=quantiles,
                                   n_to_show=n_top_patches,
                                   save_dir=save_dir,
                                   name=patient_id)
