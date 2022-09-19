"""
Sum MIL architecture from Deep Set framework

This code was built off
https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
https://github.com/KatherLab/HIA/blob/main/models/model_Attmil.py
https://github.com/mahmoodlab/PORPOISE/blob/master/models/model_attention_mil.py"""
from numbers import Number
import numpy as np

import torch.nn as nn
import torch

from var_pool.nn.arch.utils import mlp_from_tuple
from var_pool.nn.arch.VarPool import VarPool


class SumMIL(nn.Module):
    """
    Simple baseline for Deep set

    Parameters
    ----------
    encoder_dim: int
        Dimension of the encoder features. This is either the dimension output by the instance encoder (if there is one) or it is the dimension of the input feature (if there is no encoder).

    encoder: None, nn.Module
        (Optional) The bag instance encoding network.

    head: nn.Module, int, tuple of ints
        (Optional) The network after the attention mean pooling step. If an int is provided a single linear layer is added. If a tuple of ints is provided then a multi-layer perceptron with RELU activations is added.

    n_attn_latent: int, None
        Number of latent dimension for the attention layer. If None, will default to (n_in + 1) // 2.

    dropout: bool, float
        Whether or not to use dropout in the attention mechanism. If True, will default to p=0.25.

    References
    ----------
    Zaheer M. et al., Deep Sets, NeurIPS, 2017
    """

    def __init__(self, encoder_dim, encoder=None, head=None,
                 n_attn_latent=None, dropout=False):
        super().__init__()

        self.encoder = encoder

        # Setup head network
        if isinstance(head, Number):
            # single linear layer
            self.head = nn.Linear(encoder_dim, int(head))

        elif isinstance(head, (tuple, list, np.ndarray)):
            # multi layer perceptron
            dims = [encoder_dim] + list(head)
            self.head = mlp_from_tuple(dims=dims, act='relu')

        else:
            self.head = head

    def encode(self, x):
        if self.encoder is None:
            return nn.Identity(x)
        else:
            return self.encoder(x)

    def forward(self, bag):
        """
        Outputs the resutls of head.forward(x) and the attention values.

        Parameters
        -----------

        if bag is a Tensor
        bag: shape (n_batches, n_instances, *instance_dims)
            The instance bag features.

        if bag is a tuple/list
        bag: tuple, shape (2, )
            If a fixed bag size is provided then we input a tuple where the first element is the bag (as above) and the second element is the list of non_pad_size for each bag indicating how many instances in each bag are real instances and not padding.

        Output
        ------
        out: shape (n_batches, *dim_out)
            The output of the head network.
        """
        #############################################
        # compute instance embed and attention scores
        #############################################

        if isinstance(bag, (list, tuple)):
            # split input = bag, non_pad_size
            bag, non_pad_size = bag
            n_batches, n_instances = bag.shape[0:2]

        else:
            n_batches, n_instances = bag.shape[0:2]

        assert bag.ndim >= 3, "Make sure to include first batch dimension"

        # flatten batches x instances so the attention scores
        # can be easily computed in parallel. This  allows us to
        # wrap enc_and_attend a nn.DataParallel() to compute all instances
        # in parallel
        # (n_batches, n_instances, *instance_dims) ->
        # (n_batches * n_instances, *instance_dims)
        x = torch.flatten(bag, start_dim=0, end_dim=1)
        # (n_batches * n_instances, 1), (n_batches * n_instances, encode_dim)

        bag_feats = self.encoder(x)
        n_bag_feats = bag_feats.shape[1]

        # unflatten
        bag_feats = bag_feats.view(n_batches, n_instances, n_bag_feats)

        # Mean instaed of sum due to NaN issues with Cox loss (Cox exponentiates)
        # (n_batches, n_instances, encode_dim) -> (n_batches, encoder_dim)
        summed_bag_feats = bag_feats.mean(1)

        ################################
        # get output from head network #
        ################################

        if self.head is not None:
            out = self.head(summed_bag_feats)  # (n_batches, out_dim)
        else:
            out = summed_bag_feats

        return out


class SumVarPoolMIL(nn.Module):
    """
    Simple baseline for Deep set

    Parameters
    ----------
    encoder_dim: int
        Dimension of the encoder features. This is either the dimension output by the instance encoder (if there is one) or it is the dimension of the input feature (if there is no encoder).

    encoder: None, nn.Module
        (Optional) The bag instance encoding network.

    head: nn.Module, int, tuple of ints
        (Optional) The network after the attention mean pooling step. If an int is provided a single linear layer is added. If a tuple of ints is provided then a multi-layer perceptron with RELU activations is added.

    n_attn_latent: int, None
        Number of latent dimension for the attention layer. If None, will default to (n_in + 1) // 2.

    dropout: bool, float
        Whether or not to use dropout in the attention mechanism. If True, will default to p=0.25.

    References
    ----------
    Zaheer M. et al., Deep Sets, NeurIPS, 2017
    """

    def __init__(self, encoder_dim, encoder=None, head=None,
                 n_attn_latent=None, n_var_pools=100,
                 act_func='log', dropout=False):
        super().__init__()

        self.encoder = encoder

        # Variance pooling
        self.var_pool = VarPool(encoder_dim=encoder_dim,
                                n_var_pools=n_var_pools,
                                act_func=act_func,
                                apply_attn=False)

        head_input_dim = encoder_dim + n_var_pools

        # Setup head network
        if isinstance(head, Number):
            # single linear layer
            self.head = nn.Linear(head_input_dim, int(head))

        elif isinstance(head, (tuple, list, np.ndarray)):
            # multi layer perceptron
            dims = [head_input_dim] + list(head)
            self.head = mlp_from_tuple(dims=dims, act='relu')

        else:
            self.head = head

    def encode(self, x):
        if self.encoder is None:
            return nn.Identity(x)
        else:
            return self.encoder(x)

    def forward(self, bag):
        """
        Outputs the resutls of head.forward(x) and the attention values.

        Parameters
        -----------

        if bag is a Tensor
        bag: shape (n_batches, n_instances, *instance_dims)
            The instance bag features.

        if bag is a tuple/list
        bag: tuple, shape (2, )
            If a fixed bag size is provided then we input a tuple where the first element is the bag (as above) and the second element is the list of non_pad_size for each bag indicating how many instances in each bag are real instances and not padding.

        Output
        ------
        out: shape (n_batches, *dim_out)
            The output of the head network.
        """
        #############################################
        # compute instance embed and attention scores
        #############################################

        if isinstance(bag, (list, tuple)):
            # split input = bag, non_pad_size
            bag, non_pad_size = bag
            n_batches, n_instances = bag.shape[0:2]

            if all(non_pad_size == n_instances):
                # if the non-pad size of all the bags is equal to the number of
                # instances in each bag they we did not add any padding
                # e.g. each bag was subset
                is_padded_bag = False
                # this allows us to skip the uncessary masking code below
            else:
                is_padded_bag = True

        else:
            is_padded_bag = False
            n_batches, n_instances = bag.shape[0:2]

        assert bag.ndim >= 3, "Make sure to include first batch dimension"

        # (n_batches, n_instances, *instance_dims) ->
        # (n_batches * n_instances, *instance_dims)
        x = torch.flatten(bag, start_dim=0, end_dim=1)
        # (n_batches * n_instances, 1), (n_batches * n_instances, encode_dim)

        # This is a dummy attn scores
        attn_scores = torch.ones(n_batches, n_instances, 1, device=x.device)

        bag_feats = self.encoder(x)
        n_bag_feats = bag_feats.shape[1]

        # unflatten
        bag_feats = bag_feats.view(n_batches, n_instances, n_bag_feats)

        if is_padded_bag:
            # We need to set the attn score for padding instances to zero
            # and the original istances to 1/(num of original instances)
            batch_size, bag_size = bag.shape[0], bag.shape[1]

            # compute mask of true instances i.e. the non-pad elements
            instance_idxs = torch.arange(bag_size).repeat(batch_size, 1)
            instance_idxs = instance_idxs.to(attn_scores.device)

            true_instance_mask = instance_idxs < non_pad_size.unsqueeze(-1)
            true_instance_mask = true_instance_mask.unsqueeze(-1)

            # (n_batches, n_instances, 1)
            n_orig_instances = true_instance_mask.sum(1).unsqueeze(1)

            zeros = torch.zeros_like(attn_scores)
            attn = torch.where(true_instance_mask,
                               attn_scores,
                               zeros).to(attn_scores.device)

            attn = attn / n_orig_instances
        else:
            # If no zero-padding, just divide by # of instances
            attn = attn_scores / n_instances

        # Mean instaed of sum due to NaN issues with Cox loss (Cox exponentiates)
        # (n_batches, n_instances, encode_dim) -> (n_batches, encoder_dim)
        summed_bag_feats = (bag_feats * attn).sum(1)

        # Divide by n_instances to be on same scale as summed_bag_feats
        var_pooled_bag_feats = self.var_pool(bag_feats, attn)

        merged_bag_feats = \
            torch.cat((summed_bag_feats, var_pooled_bag_feats),
                      dim=1)
        # (n_batches, encode_dim +  n_var_pool)

        ################################
        # get output from head network #
        ################################

        if self.head is not None:
            out = self.head(merged_bag_feats)  # (n_batches, out_dim)
        else:
            out = merged_bag_feats

        return out
