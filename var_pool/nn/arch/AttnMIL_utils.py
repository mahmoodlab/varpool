import torch
import torch.nn as nn
import torch.nn.functional as F

from var_pool.nn.arch.AttnMIL import Attn, GatendAttn


class AttnMILMixin:
    """
    Mixin for attention MIL modules

    Parameters
    ----------
    enc_and_attend: nn.Module -> attn_scores,

    """

    def compute_bag_feats_and_attn_scores(self, bag):
        """
        Computes the instance encodings and attention scores

        Parameters
        ----------
        if bag is a Tensor
        bag: shape (n_batches, n_instances, *instance_dims)
            The instance bag features.

        if bag is a tuple/list
        bag: tuple, shape (2, )
            If a fixed bag size is provided then we input a tuple where the first element is the bag (as above) and the second element is the list of non_pad_size for each bag indicating how many instances in each bag are real instances and not padding.

        Ouput
        -----
        attn_scores, bag_feats

        attn_scores: list or torch.Tensor, (batch_size, n_instances, 1)
            The attention scores; a list if there are separate attention brancehs.

        bag_feats: (batch_size, n_insatnaces, n_bag_feats)
        """
        ################
        # Format input #
        ################
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

        #####################
        # Encode and attend #
        #####################

        attn_scores, bag_feats = self.enc_and_attend(x)
        # (n_batches * n_instances, 1), (n_batches * n_instances, encode_dim)
        n_bag_feats = bag_feats.shape[1]

        #################
        # Format output #
        #################

        # unflatten
        bag_feats = bag_feats.view(n_batches, n_instances, n_bag_feats)

        if isinstance(attn_scores, list):
            # if there are multiple attention scores
            attn_scores = [a_s.view(n_batches, n_instances, 1)
                           for a_s in attn_scores]
        else:
            attn_scores = attn_scores.view(n_batches, n_instances, 1)

        return attn_scores, bag_feats

    def computed_norm_attn(self, attn_scores,
                           is_padded_bag, non_pad_size):
        """
        Parameters
        ----------
        attn_scores: torch.Tensor (batch_size, n_instances, 1)
            The attention scores tensor.

        is_padded_bag: bool
            Whether or not the bag was padded.

        non_pad_size: array-like, shape (batch_size, ) or None

        Output
        ------
        norm_attn: (n_batches, n_instances, 1)
        """

        batch_size, n_instances = attn_scores.shape[0], attn_scores.shape[1]

        if not is_padded_bag:
            # attn = softmax over instances
            norm_attn = F.softmax(attn_scores, dim=1)
            # (n_batches, n_instances, 1)

        else:
            # make the attention scores for padding instances
            # very big negative numbers so they effectively get ignored
            # this code is borrowed from
            # https://github.com/KatherLab/HIA/blob/main/models/model_Attmil.py

            # compute mask of true instances i.e. the non-pad elements
            instance_idxs = torch.arange(n_instances).repeat(batch_size, 1)
            instance_idxs = instance_idxs.to(attn_scores.device)

            true_instance_mask = instance_idxs < non_pad_size.unsqueeze(-1)
            true_instance_mask = true_instance_mask.unsqueeze(-1)

            # array with very large negative number = tiny attention
            big_neg = torch.full_like(attn_scores, -1e10)
            attn_scores_ignore_pad = torch.where(true_instance_mask,
                                                 attn_scores,
                                                 big_neg)

            # attn = softmax over instances
            norm_attn = F.softmax(attn_scores_ignore_pad, dim=1)

        return norm_attn

    def get_pad_info(self, bag):
        """
        Parameters
        ----------
        bag: torch.Tensor or list/tuple

        Output
        ------
        is_padded_bag, non_pad_size

        is_padded_bag: bool
            Whether or not this is a padded bag.

        non_pad_size: None, torch.Tensor (batch_size)
        """
        if isinstance(bag, (list, tuple)):
            # split input = bag, non_pad_size
            bag, non_pad_size = bag

            n_instances = bag.shape[1]

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
            non_pad_size = None

        return is_padded_bag, non_pad_size


def get_attn_module(encoder_dim, n_attn_latent, dropout, gated):
    """
    Gets the attention module
    """
    if gated:
        return Attn(n_in=encoder_dim,
                    n_latent=n_attn_latent,
                    dropout=dropout)
    else:
        return GatendAttn(n_in=encoder_dim,
                          n_latent=n_attn_latent,
                          dropout=dropout)


class EncodeAndMultipleAttend(nn.Module):
    """
    An encoder that feeds into multiple parallel attention branches.

    Parameters
    ----------
    encoder: nn.Module, None
        The encoder that each samples is passed into.

    attns: list of nn.Module
        The attention branches.
    """

    def __init__(self, encoder, attns):
        super().__init__()

        self.encoder = encoder
        self.attns = nn.ModuleList(attns)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor, (n_batches, n_featues)

        Output
        ------
        attn_scores, instance_encodings

        attn_scores: list len(attns)
            The attention scores applied to each encoder.
        """
        if self.encoder is not None:
            x = self.encoder(x)

        # attention  modules output (attn_scores, x)
        attn_scores = [attn(x)[0] for attn in self.attns]

        return attn_scores, x
