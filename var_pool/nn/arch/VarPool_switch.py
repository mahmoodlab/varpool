"""
Variance Pooling + Attention based multiple instance learning architecture.
"""
import torch
import torch.nn as nn

from var_pool.nn.arch.AttnMIL_utils import AttnMILMixin, get_attn_module,\
    EncodeAndMultipleAttend
from var_pool.nn.arch.VarPool import VarPool


class AttnMeanAndVarPoolMIL_with_switch(AttnMILMixin, nn.Module):
    """
    Attention mean  and variance pooling architecture.

    Parameters
    ----------

    encoder_dim: int
        Dimension of the encoder features. This is either the dimension output by the instance encoder (if there is one) or it is the dimension of the input feature (if there is no encoder).

    encoder: None, nn.Module
        (Optional) The bag instance encoding network.

    mean_neck: None, nn.Module.
        (Optional) Function to apply to the mean pool embedding. Should output the same shape as var_neck which is then input to head.

    var_neck: None, nn.Module.
        (Optional) Function to apply to the var pool embedding. Should output the same shape as mean_neck which is then input to head.

    head: nn.Module,None
        (Optional)  The output of mean_neck and var_neck are added together then send to the head network.

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
    def __init__(self, encoder_dim, encoder=None,
                 mean_neck=None, var_neck=None, head=None,
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

        self._apply_var_pool = True

        ################
        # Head network #
        ################

        self.mean_neck = mean_neck
        self.var_neck = var_neck
        self.head = head

    def var_pool_off(self):
        """
        Turns variance pooling off.
        """
        self._apply_var_pool = False
        return self

    def var_pool_on(self):
        """
        Turns variance pooling on.
        """
        self._apply_var_pool = True
        return self

    def get_variance(self, bag, normalize=True):
        """
        Get variance contribution. Essentially truncated forward pass to obtain var_pooled_bag_feats

        Parameters
        ----------
        normalize: Bool
            Normalize each variance by norm of the projection vector

        Output
        ------
        var_pooled_bag_feats (n_batches, var_pool)
        """

        # instance encodings and attention scores
        attn_scores, bag_feats = self.compute_bag_feats_and_attn_scores(bag)

        is_padded_bag, non_pad_size = self.get_pad_info(bag)

        # normalize attetion
        if self.separate_attn:
            var_attn_scores = attn_scores[1]

            var_attn = self.computed_norm_attn(attn_scores=var_attn_scores,
                                               is_padded_bag=is_padded_bag,
                                               non_pad_size=non_pad_size)

        else:
            _attn = self.computed_norm_attn(attn_scores=attn_scores,
                                            is_padded_bag=is_padded_bag,
                                            non_pad_size=non_pad_size)

            var_attn = _attn

        var_pooled_bag_feats = self.var_pool(bag_feats, var_attn)

        if normalize:
            norm = torch.norm(self.var_pool.var_projections.weight.data, dim=1)
            var_pooled_bag_feats /= norm**2  # squared since it's variance

        return var_pooled_bag_feats

    def encode(self, x):
        return self.encoder(x)

    def forward(self, bag):

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

            if self._apply_var_pool:
                var_attn = self.computed_norm_attn(attn_scores=var_attn_scores,
                                                   is_padded_bag=is_padded_bag,
                                                   non_pad_size=non_pad_size)

            else:
                var_attn = None

        else:
            _attn = self.computed_norm_attn(attn_scores=attn_scores,
                                            is_padded_bag=is_padded_bag,
                                            non_pad_size=non_pad_size)

            mean_attn = _attn
            var_attn = _attn

        #####################
        # Attention pooling #
        #####################

        mean_pooled_feats = (bag_feats * mean_attn).sum(1)
        # (n_batches, n_instances, encode_dim) -> (n_batches, encoder_dim)
        if self.mean_neck is not None:
            mean_pooled_feats = self.mean_neck(mean_pooled_feats)
            # (n_batches, encoder_dim) -> (n_batches, cat_head_dim)

        if self._apply_var_pool:

            var_pooled_feats = self.var_pool(bag_feats, var_attn)
            # (n_batches, n_instances, encode_dim) -> (n_batches, n_var_pools)
            if self.var_neck is not None:
                var_pooled_feats = self.var_neck(var_pooled_feats)
                # (n_batches, n_var_pools) -> (n_batches, cat_head_dim)

        ################################
        # get output from head network #
        ################################

        if self._apply_var_pool:
            head_input = mean_pooled_feats + var_pooled_feats
        else:
            head_input = mean_pooled_feats
        # (n_batches, cat_head_dim)

        if self.head is not None:
            return self.head(head_input)
            # (n_batches, out_dim)
        else:
            return head_input
