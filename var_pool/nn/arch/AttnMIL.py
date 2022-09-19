"""
Attention based multiple instance learning architecture.

This code was built off
https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
https://github.com/KatherLab/HIA/blob/main/models/model_Attmil.py
https://github.com/mahmoodlab/PORPOISE/blob/master/models/model_attention_mil.py"""
from numbers import Number
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch

from var_pool.nn.arch.utils import mlp_from_tuple
# TODO: add support for masks


class AttnMeanPoolMIL(nn.Module):
    """
    Attention mean pooling architecture of (Ilse et al, 2018).

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

    gated: bool
        Use the gated attention mechanism.

    dropout: bool, float
        Whether or not to use dropout in the attention mechanism. If True, will default to p=0.25.

    References
    ----------
    Ilse, M., Tomczak, J. and Welling, M., 2018, July. Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    """
    def __init__(self, encoder_dim,  encoder=None, head=None,
                 n_attn_latent=None, gated=True, dropout=False):
        super().__init__()

        # setup attention mechanism
        if gated:
            attention = Attn(n_in=encoder_dim,
                             n_latent=n_attn_latent,
                             dropout=dropout)
        else:
            attention = GatendAttn(n_in=encoder_dim,
                                   n_latent=n_attn_latent,
                                   dropout=dropout)

        self.encoder = encoder

        # concat encoder and attention
        if encoder is not None:
            self.enc_and_attend = nn.Sequential(self.encoder, attention)
        else:
            self.enc_and_attend = attention

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

        # flatten batches x instances so the attention scores
        # can be easily computed in parallel. This  allows us to
        # wrap enc_and_attend a nn.DataParallel() to compute all instances
        # in parallel
        # (n_batches, n_instances, *instance_dims) ->
        # (n_batches * n_instances, *instance_dims)
        x = torch.flatten(bag, start_dim=0, end_dim=1)

        attn_scores, bag_feats = self.enc_and_attend(x)
        # (n_batches * n_instances, 1), (n_batches * n_instances, encode_dim)
        n_bag_feats = bag_feats.shape[1]

        # unflatten
        bag_feats = bag_feats.view(n_batches, n_instances, n_bag_feats)
        attn_scores = attn_scores.view(n_batches, n_instances, 1)

        # calculate attention
        if not is_padded_bag:
            # attn = softmax over instances
            attn = F.softmax(attn_scores, dim=1)
            # (n_batches, n_instances, 1)

        else:
            # make the attention scores for padding instances
            # very big negative numbers so they effectively get ignored
            # this code is borrowed from
            # https://github.com/KatherLab/HIA/blob/main/models/model_Attmil.py
            batch_size, bag_size = bag.shape[0], bag.shape[1]

            # compute mask of true instances i.e. the non-pad elements
            instance_idxs = torch.arange(bag_size).repeat(batch_size, 1)
            instance_idxs = instance_idxs.to(attn_scores.device)

            true_instance_mask = instance_idxs < non_pad_size.unsqueeze(-1)
            true_instance_mask = true_instance_mask.unsqueeze(-1)

            # array with very large negative number = tiny attention
            big_neg = torch.full_like(attn_scores, -1e10)
            attn_scores_ignore_pad = torch.where(true_instance_mask,
                                                 attn_scores,
                                                 big_neg)

            # attn = softmax over instances
            attn = F.softmax(attn_scores_ignore_pad, dim=1)

        ##########################
        # attention mean pooling #
        ##########################

        # (n_batches, n_instances, encode_dim) -> (n_batches, encoder_dim)
        weighted_avg_bag_feats = (bag_feats * attn).sum(1)

        ################################
        # get output from head network #
        ################################

        if self.head is not None:
            out = self.head(weighted_avg_bag_feats)  # (n_batches, out_dim)
        else:
            out = weighted_avg_bag_feats

        return out


class Attn(nn.Module):
    """
    The attention mechanism from Equation (8) of (Ilse et al, 2008).

    Parameters
    ----------
    n_in: int
        Number of input dimensions.

    n_latent: int, None
        Number of latent dimensions. If None, will default to (n_in + 1) // 2.

    dropout: bool, float
        Whether or not to use dropout. If True, will default to p=0.25

    References
    ----------
    Ilse, M., Tomczak, J. and Welling, M., 2018, July. Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    """

    def __init__(self, n_in, n_latent=None, dropout=False):
        super().__init__()

        if n_latent is None:
            n_latent = (n_in + 1) // 2

        # basic attention scoring module
        self.score = [nn.Linear(n_in, n_latent),
                      nn.Tanh(),
                      nn.Linear(n_latent, 1)]

        # maybe add dropout
        if dropout:
            if isinstance(dropout, Number):
                p = dropout
            else:
                p = 0.25
            self.score.append(nn.Dropout(p))

        self.score = nn.Sequential(*self.score)

    def forward(self, x):
        """
        Outputs normalized attention.

        Parameters
        -----------
        x: shape (n_batches, n_instances, n_in) or (n_instances, n_in)
            The bag features.

        Output
        ------
        attn_scores, x

        attn_scores: shape (n_batches, n_instances, 1) or (n_insatnces, 1)
            The unnormalized attention scores.

        x:
            The input tensor.
        """
        attn_scores = self.score(x)
        # attn = F.softmax(attn_scores, dim=1)  # softmax over instances

        return attn_scores, x


class GatendAttn(nn.Module):
    """
    The gated attention mechanism from Equation (9) of (Ilse et al, 2008).

    Parameters
    ----------
    n_in: int
        Number of input dimensions.

    n_latent: int, None
        Number of latent dimensions. If None, will default to (n_in + 1) // 2.

    dropout: bool, float
        Whether or not to use dropout. If True, will default to p=0.25

    References
    ----------
    Ilse, M., Tomczak, J. and Welling, M., 2018, July. Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    """

    def __init__(self, n_in, n_latent=None, dropout=False):
        super().__init__()

        if n_latent is None:
            n_latent = (n_in + 1) // 2

        self.tanh_layer = [nn.Linear(n_in, n_latent),
                           nn.Tanh()]

        self.sigmoid_layer = [nn.Linear(n_in, n_latent),
                              nn.Sigmoid()]

        # maybe add dropout
        if dropout:
            if isinstance(dropout, Number):
                p = dropout
            else:
                p = 0.25

            self.tanh_layer.append(nn.Dropout(p))
            self.sigmoid_layer.append(nn.Dropout(p))

        self.tanh_layer = nn.Sequential(*self.tanh_layer)
        self.sigmoid_layer = nn.Sequential(*self.sigmoid_layer)

        self.w = nn.Linear(n_latent, 1)

    def forward(self, x):
        """
        Outputs normalized attention.

        Parameters
        -----------
        x: shape (n_batches, n_instances, n_in) or (n_instances, n_in)
            The bag features.

        Output
        ------
        attn_scores, x

        attn_scores: shape (n_batches, n_instances, 1) or (n_insatnces, 1)
            The unnormalized attention scores.

        x:
            The input tensor.
        """

        attn_scores = self.w(self.tanh_layer(x) * self.sigmoid_layer(x))
        # attn = F.softmax(attn_scores, dim=1)  # softmax over instances

        return attn_scores, x
