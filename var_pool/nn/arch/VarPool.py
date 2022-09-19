"""
Variance Pooling + Attention based multiple instance learning architecture.
"""
import torch
import torch.nn as nn

from numbers import Number
import numpy as np

from var_pool.nn.arch.AttnMIL_utils import AttnMILMixin, get_attn_module,\
    EncodeAndMultipleAttend
from var_pool.nn.arch.utils import mlp_from_tuple


class AttnMeanAndVarPoolMIL(AttnMILMixin, nn.Module):
    """
    Attention mean  and variance pooling architecture.

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

    separate_attn: bool
        WHether or not we want to use separate attention branches for the mean and variance pooling.

    n_var_pools: int
        Number of variance pooling projections.

    act_func: str
        The activation function to apply to variance pooling. Must be one of ['sqrt', 'log', 'sigmoid'].

    log_eps: float
        Epsilon value for log(epsilon + ) var pool activation function.

    dropout: bool, float
        Whether or not to use dropout in the attention mechanism. If True, will default to p=0.25.

    References
    ----------
    Ilse, M., Tomczak, J. and Welling, M., 2018, July. Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    """
    def __init__(self, encoder_dim, encoder=None, head=None,
                 n_attn_latent=None, gated=True,
                 separate_attn=False, n_var_pools=100,
                 act_func='sqrt', log_eps=0.01,
                 dropout=False):
        super().__init__()

        ###########################
        # Setup encode and attend #
        ###########################
        self.separate_attn = bool(separate_attn)
        self.encoder = encoder

        if self.separate_attn:
            attns = [get_attn_module(encoder_dim=encoder_dim,
                                     n_attn_latent=n_attn_latent,
                                     dropout=dropout,
                                     gated=gated)
                     for _ in range(2)]

            self.enc_and_attend = EncodeAndMultipleAttend(encoder=self.encoder,
                                                          attns=attns)

        else:
            attention = get_attn_module(encoder_dim=encoder_dim,
                                        n_attn_latent=n_attn_latent,
                                        dropout=dropout,
                                        gated=gated)

            if encoder is not None:
                self.enc_and_attend = nn.Sequential(self.encoder, attention)
            else:
                self.enc_and_attend = attention

        ####################
        # Variance pooling #
        ####################
        self.var_pool = VarPool(encoder_dim=encoder_dim,
                                n_var_pools=n_var_pools,
                                log_eps=log_eps,
                                act_func=act_func)

        ################
        # Head network #
        ################

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

    def get_encode_and_attend(self, bag):

        ###################################
        # Instance encoding and attention #
        ###################################

        # instance encodings and attention scores
        attn_scores, bag_feats = self.compute_bag_feats_and_attn_scores(bag)

        is_padded_bag, non_pad_size = self.get_pad_info(bag)

        # normalize attetion
        if self.separate_attn:
            mean_attn_scores = attn_scores[0]
            var_attn_scores = attn_scores[1]

            mean_attn = self.computed_norm_attn(attn_scores=mean_attn_scores,
                                                is_padded_bag=is_padded_bag,
                                                non_pad_size=non_pad_size)

            var_attn = self.computed_norm_attn(attn_scores=var_attn_scores,
                                               is_padded_bag=is_padded_bag,
                                               non_pad_size=non_pad_size)

        else:
            _attn = self.computed_norm_attn(attn_scores=attn_scores,
                                            is_padded_bag=is_padded_bag,
                                            non_pad_size=non_pad_size)

            mean_attn = _attn
            var_attn = _attn

        return bag_feats, mean_attn, var_attn

    def forward(self, bag):

        bag_feats, mean_attn, var_attn = self.get_encode_and_attend(bag)

        #####################
        # Attention pooling #
        #####################

        # (batch_size, n_instances, encode_dim) -> (batch_size, encoder_dim)
        weighted_avg_bag_feats = (bag_feats * mean_attn).sum(1)

        var_pooled_bag_feats = self.var_pool(bag_feats, var_attn)
        # (batch_size, n_var_pool)

        ################################
        # get output from head network #
        ################################
        merged_bag_feats = \
            torch.cat((weighted_avg_bag_feats, var_pooled_bag_feats),
                      dim=1)
        # (batch_size, encode_dim +  n_var_pool)

        if self.head is not None:
            return self.head(merged_bag_feats)
            # (batch_size, out_dim)
        else:
            return merged_bag_feats


class VarPool(nn.Module):
    """
    A variance pooling layer.

    Compute the variance across attended & projected instances

    Parameters
    ----------
    encoder_dim: int
        Dimension of the encoder features.

    n_var_pools: int
        Number of variance pooling projections.

    act_func: str
        The activation function to apply to variance pooling. Must be on of ['sqrt', 'log', 'sigmoid', 'identity'].

    log_eps: float
        Epsilon value for log(epsilon + ).

    apply_attn: bool
        If True, apply attn to var projection. If False, do not apply attn (Mainly for SumMIL)

    """
    def __init__(self, encoder_dim, n_var_pools, act_func='sqrt', log_eps=0.01):
        super().__init__()
        assert act_func in ['sqrt', 'log', 'sigmoid', 'identity']

        self.var_projections = nn.Linear(encoder_dim, int(n_var_pools),
                                         bias=False)
        self.act_func = act_func
        self.log_eps = log_eps

    def init_var_projections(self):
        """
        Initializes the variance projections from isotropic gaussians such that each projections expected norm is 1
        """
        encoder_dim, n_pools = self.var_projections.weight.data.shape

        self.var_projections.weight.data = \
            torch.normal(mean=torch.zeros(encoder_dim, n_pools),
                         std=1/np.sqrt(encoder_dim))

    # def get_projection_vector(self, idx):
    #     """
    #     Returns a projection vector
    #     """
    #     return self.var_projections.weight.data[idx, :].detach()

    def get_proj_attn_weighted_resids_sq(self, bag, attn, return_resids=False):
        """
        Computes the attention weighted squared residuals of each instance to the projection mean.

        Parameters
        ----------
        bag: (batch_size, n_instances, instance_dim)
            The bag features.

        attn: (batch_size, n_instances, 1)
            The normalized instance attention scores.

        Output
        ------
        attn_resids_sq: (batch_size, n_instances, n_var_pools)
            The attention weighted squared residuals.

        if return_resids is True then we also return resids

        """
        assert len(bag.shape) == 3, \
            "Be sure to include batch in first dimension"

        projs = self.var_projections(bag)
        # (batch_size, n_instances, n_var_pools)

        if attn is None:
            attn = 1 / projs.shape[1]

        proj_weighted_avg = (projs * attn).sum(1)
        # (batch_size, n_var_pools)

        resids = projs - proj_weighted_avg.unsqueeze(1)
        attn_resids_sq = attn * (resids ** 2)
        # (batch_size, n_instances, n_var_pools)

        if return_resids:
            return attn_resids_sq, resids
        else:
            return attn_resids_sq

    def forward(self, bag, attn):
        """
        Parameters
        ----------
        bag: (batch_size, n_instances, instance_dim)
            The bag features.

        attn: (batch_size, n_instances, 1)
            The normalized instance attention scores.

        Output
        ------
        var_pool: (batch_size, n_var_pools)
        """
        attn_resids_sq = self.\
            get_proj_attn_weighted_resids_sq(bag, attn, return_resids=False)
        # (batch_size, n_instances, n_var_pools)

        # computed weighted average -- note this effectively uses
        # denominator 1/n since the attn sum to one.
        var_pool = (attn_resids_sq).sum(1)
        # (batch_size, n_var_pools)

        if self.act_func == 'sqrt':
            return torch.sqrt(var_pool)

        elif self.act_func == 'log':
            return torch.log(self.log_eps + var_pool)

        elif self.act_func == 'sigmoid':
            return torch.sigmoid(var_pool)

        elif self.act_func == 'identity':
            return var_pool
