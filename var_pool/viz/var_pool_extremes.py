import pandas as pd
import numpy as np
import os
from itertools import product
from openslide import open_slide
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from var_pool.file_utils import get_file_names, join_and_make
from var_pool.viz.top_attn import load_patient_patch_data
from var_pool.viz.utils import read_region, make_image_grid,\
    get_idxs_largest, get_idxs_smallest, save_fig


def viz_top_var_proj_patches(model, wsi_fpath, h5_fpath, autogen_fpath,
                             n_top_patches=10,
                             with_attn=True,
                             device=None,
                             save_dir=None,
                             fname_stub=None):
    """
    Visualizes the most extreme patches on either end of the variance pool projections.

    Parameteres
    -----------
    model: nn.Module
        The model.

    wsi_fpath: str, list or str
        Path(s) to WSI image.

    h5_fpath: str, list of str
        Path(s) to hdf5 file containing the features and coordinates.

    autogen_fpath: str.
        Path to autogen file containint image metadata needed to load the patches.

    with_attn: bool
        Whether or not to incorperate attention when computing the residuals. The model uses attention to make its predictions.

    device: None, device
        The device to put the tensors on.

    save_dir: None, str
        (Optional) Where to save the images.

    save_fpath_stub: None, str
        (Optional) File path stub for saving the top patch images. The files will be saved as save_dir/proj_J/save_fpath_stub-PROJ-SIGN.png where J is the projection index and SIGN is either pos/neg.

    Output
    ------
    top_patches: dict of list of PIL Imagages
        The top positive/negative patches for each variance pool projection.
    """
    assert sum([save_dir is None, fname_stub is None]) in [0, 2],\
        "Both save_dir and save_fpath_stub need to be provided"

    autogen = pd.read_csv(autogen_fpath, index_col='slide_id')
    feats, coords, patch_ids = load_patient_patch_data(wsi_fpath, h5_fpath)
    model.eval()

    ################################
    # Compute variance projections #
    ################################
    bag = torch.from_numpy(feats).unsqueeze(0)
    if device is not None:
        bag = bag.to(device)
    with torch.no_grad():
        # get attention weighted residuals
        bag_feats, mean_attn, var_attn = model.get_encode_and_attend(bag)

        if not with_attn:
            var_attn = None

        attn_resids_sq, resids = model.var_pool.\
            get_proj_attn_weighted_resids_sq(bag_feats, var_attn,
                                             return_resids=True)

        attn_resids_sq = attn_resids_sq.detach().cpu().numpy().squeeze()
        resids = resids.detach().cpu().numpy().squeeze()

        # Computed signed residual
        signs = np.ones_like(attn_resids_sq)
        signs[resids < 0] = -1
        attn_resids_sq *= signs
        # (n_instances, n_pools)

    n_var_pools = attn_resids_sq.shape[1]

    ###############################
    # Determine extreme instances #
    ###############################
    top_idxs = {sign: [] for sign in ['pos', 'neg']}

    for pool_idx in range(n_var_pools):

        # signed variance projections
        this_signed_attn_resid_sq = attn_resids_sq[:, pool_idx]

        idxs_pos = get_idxs_largest(this_signed_attn_resid_sq, k=n_top_patches)
        idxs_pos = idxs_pos[::-1]  # largest goes first
        idxs_neg = get_idxs_smallest(this_signed_attn_resid_sq, k=n_top_patches)

        top_idxs['pos'].append(idxs_pos)
        top_idxs['neg'].append(idxs_neg)

    #################################
    # Get patches for top instances #
    #################################

    top_patches = {sign: [[] for _ in range(n_var_pools)]
                   for sign in ['pos', 'neg']}

    for pool_idx, sign, rank in product(range(n_var_pools),
                                        ['pos', 'neg'],
                                        range(n_top_patches)
                                        ):

        # get info for this patch
        patient_level_idx = top_idxs[sign][pool_idx][rank]

        # get the WSI/patch coordinate for this patch
        wsi_idx, patch_idx = patch_ids[patient_level_idx]

        wsi = open_slide(wsi_fpath[wsi_idx])

        # metadata needed to load the patches
        wsi_fname = get_file_names(wsi_fpath[wsi_idx]) + '.svs'
        patch_level, patch_size, custom_downsample = \
            autogen.loc[wsi_fname]\
            [['patch_level', 'patch_size', 'custom_downsample']]

        location = coords[wsi_idx][patch_idx, :]

        patch = read_region(wsi=wsi,
                            location=location,
                            level=patch_level,
                            patch_size=patch_size,
                            custom_downsample=custom_downsample)

        top_patches[sign][pool_idx].append(patch)

    ###################
    # Make patch grid #
    ###################

    for pool_idx, sign in product(range(n_var_pools),
                                  ['pos', 'neg']
                                  ):

        image_grid = make_image_grid(top_patches[sign][pool_idx],
                                     pad=3, n_cols=10)
        top_patches[sign][pool_idx] = image_grid

        if fname_stub is not None:

            # maybe make save dir
            proj_save_dir = join_and_make(save_dir, 'proj_{}'.format(pool_idx))

            save_fpath = os.path.join(proj_save_dir,
                                      '{}-{}.png'.format(fname_stub, sign)
                                      )

            image_grid.save(save_fpath)

    return top_patches


def viz_var_proj_patches_quantiles(model, wsi_fpath, h5_fpath, autogen_fpath,
                                   save_dir,
                                   name,
                                   n_to_show=10,
                                   quantiles=[0, 25, 50, 75, 100],
                                   with_attn=True,
                                   device=None):
    """
    Visualizes the most extreme patches on either end of the variance pool projections.

    Parameteres
    -----------
    model: nn.Module
        The model.

    wsi_fpath: str, list or str
        Path(s) to WSI image.

    h5_fpath: str, list of str
        Path(s) to hdf5 file containing the features and coordinates.

    autogen_fpath: str.
        Path to autogen file containint image metadata needed to load the patches.

    save_dir: str
        Where to save the images.

    name: str
         Name of this patient. The files will be saved as save_dir/proj_J/save_fpath_stub-PROJ-name.png where J is the projection index .

    n_to_show: int
        Number of patches to show for each quantile.

    with_attn: bool
        Whether or not to incorperate attention when computing the residuals. The model uses attention to make its predictions.

    device: None, device
        The device to put the tensors on.
    """
    autogen = pd.read_csv(autogen_fpath, index_col='slide_id')
    feats, coords, patch_ids = load_patient_patch_data(wsi_fpath, h5_fpath)
    model.eval()

    ################################
    # Compute variance projections #
    ################################
    bag = torch.from_numpy(feats).unsqueeze(0)
    if device is not None:
        bag = bag.to(device)
    with torch.no_grad():
        # get attention weighted residuals
        bag_feats, mean_attn, var_attn = model.get_encode_and_attend(bag)

        if not with_attn:
            var_attn = None

        attn_resids_sq, resids = model.var_pool.\
            get_proj_attn_weighted_resids_sq(bag_feats, var_attn,
                                             return_resids=True)

        attn_resids_sq = attn_resids_sq.detach().cpu().numpy().squeeze()
        resids = resids.detach().cpu().numpy().squeeze()

        # Computed signed residual
        signs = np.ones_like(attn_resids_sq)
        signs[resids < 0] = -1
        attn_resids_sq *= signs
        # (n_instances, n_pools)

    n_var_pools = attn_resids_sq.shape[1]

    ###########################################
    # Plot histograph of var pool projections #
    ###########################################
    for pool_idx in range(n_var_pools):

        # this signed attention SQUARED residuals
        this_arsq = attn_resids_sq[:, pool_idx]

        # this signed attnetion residuals
        this_ar = np.sign(this_arsq) * np.sqrt(abs(this_arsq))
        quantile_values = np.percentile(this_ar, q=quantiles)

        # variance for this pool
        this_var = np.sum(abs(this_arsq))

        plt.figure(figsize=(8, 8))
        # plt.hist(this_resids, bins=50)
        sns.histplot(this_ar, bins=100)
        sns.rugplot(this_ar, color='red')
        plt.xlabel("Var projection {}, attention mean residuals".
                   format(pool_idx))
        plt.title("Variance = {}".format(this_var))
        plt.axvline(0, color='black')
        for i in range(len(quantiles)):
            plt.axvline(quantile_values[i], ls='--', alpha=.5,
                        color='grey',
                        label='quantile {:1.0f}, ({})'.
                              format(quantiles[i], quantile_values[i]))
        plt.legend()
        stats_save_dir = join_and_make(save_dir,
                                       'proj_{}'.format(pool_idx),
                                       'histograms')
        save_fpath = os.path.join(stats_save_dir,
                                  '{}_histogram.png'.format(name)
                                  )

        save_fig(save_fpath)

    ##########################################
    # Correlation structure pool projections #
    ##########################################
    corr_save_dir = join_and_make(save_dir,
                                  'proj_correlation')
    save_fpath = os.path.join(corr_save_dir,
                              '{}_corr.png'.format(name)
                              )
    plot_corr(attn_resids_sq, method='spearman')
    save_fig(save_fpath)

    ###################################
    # Make patch viz for each var pool #
    ####################################

    quantiles = np.sort(quantiles)
    n_tot_patches = attn_resids_sq.shape[0]

    for pool_idx in range(n_var_pools):

        # signed variance projections for this variance pool
        this_resids = attn_resids_sq[:, pool_idx]

        # sort for easy access to quantiles
        this_resids = pd.Series(this_resids)
        this_resids = this_resids.\
            sort_values(ascending=True)  # index is patch identifier

        ################################
        # make image for each quantile #
        ################################
        image_this_pool = []
        for quant in quantiles:

            # get patches for this quantile
            idx_start, idx_end = \
                get_idxs_around_quantile(quantile=quant,
                                         n_tot=n_tot_patches,
                                         n_to_show=n_to_show)

            patch_idxs_to_show = \
                this_resids.index.values[idx_start:idx_end]

            this_quantile_patches = []
            for patient_level_idx in patch_idxs_to_show:

                # get the WSI/patch coordinate for this patch
                wsi_idx, patch_idx = patch_ids[patient_level_idx]

                wsi = open_slide(wsi_fpath[wsi_idx])

                # metadata needed to load the patches
                wsi_fname = get_file_names(wsi_fpath[wsi_idx]) + '.svs'
                patch_level, patch_size, custom_downsample = \
                    autogen.loc[wsi_fname]\
                    [['patch_level', 'patch_size', 'custom_downsample']]

                location = coords[wsi_idx][patch_idx, :]

                patch = read_region(wsi=wsi,
                                    location=location,
                                    level=patch_level,
                                    patch_size=patch_size,
                                    custom_downsample=custom_downsample)

                this_quantile_patches.append(patch)

            image_this_pool.append(make_image_grid(this_quantile_patches,
                                                   pad=3, n_cols=10)
                                   )

        ############################
        # Save image for this pool #
        ############################
        image_this_pool = make_image_grid(image_this_pool, pad=100, n_cols=1)

        proj_save_dir = join_and_make(save_dir, 'proj_{}'.format(pool_idx))
        save_fpath = os.path.join(proj_save_dir,
                                  '{}.png'.format(name)
                                  )

        image_this_pool.save(save_fpath)


def plot_corr(df, method='pearson'):
    """
    Makes visualizations of the the correlation matrix.

    Parameters
    ----------
    df: pd.DataFrame, shape (n_samples, n_features)
        The data matrix.

    method: str
        Which correlation method to use; see df.corr(method=method).
    """

    # TODO: automatically infer this from number of features
    figsize = (16, 16)

    df = pd.DataFrame(df)

    # Visualize correlation matrix
    plt.figure(figsize=figsize)
    corr = df.corr(method=method)
    sns.heatmap(corr, center=0, vmin=-1, vmax=1, cmap='RdBu',
                annot=True, fmt=".3f")
    plt.title('{} correlation'.format(method))
    plt.xticks(np.arange(corr.shape[0]), corr.columns.values)
    plt.yticks(np.arange(corr.shape[0]), corr.columns.values)


def get_idxs_around_quantile(quantile, n_tot, n_to_show):
    # patches for each quantile
    if quantile == 0:
        # first set of patches
        idx_start = 0
        idx_end = n_to_show

    elif quantile == 100:
        # last set of patches
        idx_end = n_tot
        idx_start = idx_end - n_to_show

    else:
        # patches on either side of this quantile
        idx_middle = int((quantile/100) * n_tot)
        idx_start = idx_middle - n_to_show // 2
        idx_end = idx_start + n_to_show

        # make sure endge case works out
        idx_start = max(idx_start, 0)
        idx_end = min(idx_end, n_tot)

    return idx_start, idx_end
