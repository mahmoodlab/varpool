import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import os
import torch
from openslide import open_slide

from var_pool.file_utils import get_file_names
from var_pool.viz.utils import read_region, make_image_grid, get_idxs_largest


def viz_top_attn_patches(model, wsi_fpath, h5_fpath, autogen_fpath,
                         device=None,
                         n_top_patches=10, save_fpath=None):
    """
    Visualizes the top attended patches. Handles the case when a patient has multiple WSIs.

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

    device: None, device
        The device to put the tensors on.

    save_fpath: None, str
        (Optional) Path to save the top attneded patch grid.

    Output
    ------
    top_attn_patches: PIL.Image
        The top attended patches concatenated together in a grid.
    """

    autogen = pd.read_csv(autogen_fpath, index_col='slide_id')
    feats, coords, patch_ids = load_patient_patch_data(wsi_fpath, h5_fpath)
    model.eval()

    # Compute attention scores
    bag = torch.from_numpy(feats).unsqueeze(0)
    if device is not None:
        bag = bag.to(device)
    with torch.no_grad():
        attn_scores, _ = model.enc_and_attend(bag)
        attn_scores = attn_scores.detach().cpu().numpy().squeeze()

    # get top attended patches
    idxs_top_attn = get_idxs_largest(attn_scores, k=n_top_patches)

    # Load each of the top patches
    top_patches = []
    for patient_level_idx in idxs_top_attn:

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

        top_patches.append(patch)

    # concatenated top patches together
    top_patches = make_image_grid(top_patches, pad=3, n_cols=10)

    # maybe save the image
    if save_fpath is not None:
        save_dir = Path(save_fpath).parent
        os.makedirs(save_dir, exist_ok=True)

        top_patches.save(save_fpath)

    return top_patches


def load_patient_patch_data(wsi_fpath, h5_fpath):
    """
    Loads the patch features and coordinates for a given patient who may have multiple WSIs.

    Parameters
    ----------
    wsi_fpath: str, list or str
        Path(s) to WSI image.

    h5_fpath: str, list of str
        Path(s) to hdf5 file containing the features and coordinates.

    Output
    -----
    feats, coords, patch_ids

    feats: array like, (n_patches_tot, n_feats)
        The patch features for each WSI.

    coords: list of array-like
        The patch coordinates for each WSI.

    patch_ids: array-like, (n_patches_tot, 2)
        The first column identifies which WSI the patch belongs to, the second column identifies which patch it is.
    """

    # ensure wsi_fpath/h5_fpath are lists of str
    if isinstance(wsi_fpath, str):
        wsi_fpath = [wsi_fpath]
        h5_fpath = [h5_fpath]
    assert len(wsi_fpath) == len(h5_fpath)
    n_wsis = len(wsi_fpath)

    # Load feats/coords for each WSI
    coords = []
    feats = []
    patch_ids = []
    for wsi_idx in range(n_wsis):

        # Load coords/instance features for this image
        with h5py.File(h5_fpath[wsi_idx], 'r') as hdf5_file:
            coords.append(np.array(hdf5_file['coords']))
            feats.append(np.array(hdf5_file['features']))

            # patch_id = (wsi_idx, patch_idx)
            n_patches = hdf5_file['features'].shape[0]
            patch_ids.append(np.vstack([np.repeat(wsi_idx, repeats=n_patches),
                                        np.arange(n_patches)]).T)

    patch_ids = np.vstack(patch_ids)
    feats = np.vstack(feats)

    return feats, coords, patch_ids
