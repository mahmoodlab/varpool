import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np


def to_fixed_size_bag(bag, fixed_bag_size=512):
    """
    Takes an input bag and returns a bag of a fixed size through either random subsampling or zero padding.
    The random sampling is always done with torch so if you want reproducible results make sure to set the torch seed ahead of time.

    Parameters
    ----------
    bag: torch.Tensor
        The bag of samples.

    fixed_bag_size: int
        Size of the desired bag.

    Output
    ------
    new_bag, non_pad_size

    new_bag:
        The padded or subsampled bag with a fixed size.

    non_pad_size: int
        The number of samples in the bag corresponding to non padding elements i.e. this is min(bag_size, len(bag))
    """
    # borrowed from  https://github.com/KatherLab/HIA/blob/main/models/model_Attmil.py
    fixed_bag_size = int(fixed_bag_size)
    this_bag_size = bag.shape[0]

    if this_bag_size > fixed_bag_size:

        # randomly subsample instances
        bag_idxs = torch.randperm(bag.shape[0])  # random smaple using torch
        bag_idxs = list(bag_idxs.numpy())
        bag_idxs = bag_idxs[:fixed_bag_size]

        new_bag = bag[bag_idxs]

        non_pad_size = fixed_bag_size

    elif this_bag_size < fixed_bag_size:
        # zero-pad if we don't have enough samples
        n_feats = bag.shape[1]
        n_to_pad = fixed_bag_size - this_bag_size
        pad_size = (n_to_pad, n_feats)

        if isinstance(bag, torch.Tensor):
            pad = torch.zeros(pad_size,
                              dtype=bag.dtype,
                              device=bag.device)

            new_bag = torch.cat((bag, pad))

        else:
            pad = np.zeros(pad_size, dtype=bag.dtype)
            new_bag = np.vstack([bag, pad])

        non_pad_size = this_bag_size

    else:
        new_bag = bag
        non_pad_size = this_bag_size

    return new_bag, non_pad_size


def get_collate_fixed_bag_size(fixed_bag_size):
    """
    Gets the collate_fixed_bag_size function which sets every bag in the batch to a fixed number of instances (via random subsampling or padding) then calls torch.utils.data._utils.collate.default_collate.

    Note each bag array is replaced with a tuple of (bag, non_pad_size) where non_pad_size is as in to_fixed_size_bag.

    Parameters
    ----------
    fixed_bag_size: int, str, None
        If an int, the size of the desired bag. If set to 'max', then will use the size of the largest bag in the batch.

    Output
    ------
    collate_fixed_bag_size: callable
    """

    def collate_fixed_bag_size(batch):
        """
        Parameters
        ----------
        batch: list
            The list of items in each batch. Each entry is either a an array (the bag features) or a tuple of length 2 (bag features and a response).

        Output
        ------
        The output of default_collate(batch), but we have first applied to_fixed_size_bag() for each bag and replaced each bag with a tuple of (bag, non_pad_size).
        """

        # setup bag size for this batch
        if fixed_bag_size == 'max':
            # get the largest bag size in the batch
            bag_sizes = []
            for item in batch:
                if isinstance(item, tuple):
                    bs = item[0].shape[0]
                else:
                    bs = item.shape[0]
                bag_sizes.append(bs)

            FBS = max(bag_sizes)
        else:
            FBS = fixed_bag_size

        # reformat each bag to be a tuple of (new_bag, non_pad_size)
        batch_size = len(batch)
        for i in range(batch_size):

            # pull out the bag
            if isinstance(batch[i], tuple):
                # if the items are tuples, the bag should be the first entry
                bag = batch[i][0]
            else:
                bag = batch[i]

            # make the new bag with a fixed size
            new_bag, non_pad_size = \
                to_fixed_size_bag(bag=bag, fixed_bag_size=FBS)

            # replace the old bag with tuple (new_bag, non_pad_size)
            new_item = (new_bag, non_pad_size)
            if isinstance(batch[i], tuple):
                batch[i] = (new_item, *batch[i][1:])
            else:

                batch[i] = new_item

        return default_collate(batch)

    return collate_fixed_bag_size
