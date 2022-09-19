import torch.nn as nn
from warnings import warn

from var_pool.nn.arch.AttnMIL import AttnMeanPoolMIL
from var_pool.nn.arch.VarPool import AttnMeanAndVarPoolMIL
from var_pool.nn.arch.SumMIL import SumMIL, SumVarPoolMIL
from var_pool.nn.arch.PatchGCN import PatchGCN, PatchGCN_varpool, MIL_Graph_FC, MIL_Graph_FC_varpool


def get_model_with_nn_layers(n_bag_feats, out_dim, dropout,
                             final_nonlin='relu',
                             head_n_hidden_layers=1,
                             attn_latent_dim=256,
                             head_hidden_dim=256):
    """
    Gets the attention mean pooling model where there are neural network layers before and after the attention mean pool  i.e.

    - a one layer NN instance encoder
    - gatteed attention mean pooling
    - a one layer NN head that transforms the features after attention mean poolilng

    This is the attention MIL archetecture from (Chen et al, 2021) i.e. we translate https://github.com/mahmoodlab/Patch-GCN.

    Parameters
    ----------
    n_bag_feats: int

    out_dim: int

    dropout: bool

    final_nonlin: str

    head_n_hidden_layers: int

    attn_latent_dim: int

    head_hidden_dim: int

    Output
    ------
    nn.Module

    References
    ----------
    Chen, R.J., Lu, M.Y., Shaban, M., Chen, C., Chen, T.Y., Williamson, D.F. and Mahmood, F., 2021, September. Whole Slide Images are 2D Point Clouds: Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 339-349). Springer, Cham.

    Ilse, M., Tomczak, J. and Welling, M., 2018, July. Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    """
    # https://github.com/mahmoodlab/Patch-GCN/blob/c6455a3a01c4ca20cde6ddb9a6f9cd807253a4f7/models/model_set_mil.py#L68

    encoder_dim = 512

    # additional MLP encoder for the instance features
    instance_encoder = [nn.Linear(n_bag_feats, encoder_dim), nn.ReLU()]
    if dropout:
        instance_encoder.append(nn.Dropout(p=0.25))
    instance_encoder = nn.Sequential(*instance_encoder)

    # Head transfomrs the mean pool
    head = _get_head_network(encoder_dim=encoder_dim,
                             out_dim=out_dim,
                             dropout=dropout,
                             head_n_hidden_layers=head_n_hidden_layers,
                             head_hidden_dim=head_hidden_dim,
                             final_nonlin=final_nonlin)

    return AttnMeanPoolMIL(encoder_dim=encoder_dim,
                           encoder=instance_encoder,
                           head=head,
                           n_attn_latent=attn_latent_dim,
                           gated=True,
                           dropout=dropout)


def get_model_slim(n_bag_feats, out_dim, dropout, final_nonlin='relu',
                   attn_latent_dim=256):
    """
    Gets the slimest possible attention mean pooling model
    - raw instance embeddings (no encoder)
    - attention mean pool layer
    - single linear layer after mean pooling


    Parameters
    ----------
    n_bag_feats: int

    out_dim: int

    dropout: bool

    final_nonlin: str

    attn_latent_dim: int

    Output
    ------
    nn.Module

    References
    -----------
    Ilse, M., Tomczak, J. and Welling, M., 2018, July. Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    """

    return AttnMeanPoolMIL(encoder_dim=n_bag_feats,
                           encoder=None,
                           head=nn.Linear(n_bag_feats, out_dim),
                           n_attn_latent=attn_latent_dim,
                           gated=True,
                           dropout=dropout)


def get_model_varpool(n_bag_feats,
                      out_dim,
                      n_var_pools,
                      var_act_func,
                      separate_attn,
                      dropout,
                      final_nonlin='relu',
                      head_n_hidden_layers=1,
                      attn_latent_dim=256,
                      head_hidden_dim=256):
    """
    Create a Varpool + attention pool model.


    Parameters
    ----------
    n_bag_feats: int

    out_dim: int

    n_var_pools: int

    var_act_func: str

    separate_attn: bool

    dropout: bool

    final_nonlin: str

    head_n_hidden_layers: int

    attn_latent_dim: int

    head_hidden_dim: int

    Output
    ------
    nn.Module


    """
    encoder_dim = 512
    # additional MLP encoder for the instance features
    instance_encoder = [nn.Linear(n_bag_feats, encoder_dim), nn.ReLU()]
    if dropout:
        instance_encoder.append(nn.Dropout(p=0.25))
    instance_encoder = nn.Sequential(*instance_encoder)

    # head transforms the concatenation of the mean pool and variance pool
    head = _get_head_network(encoder_dim=encoder_dim + n_var_pools,
                             out_dim=out_dim,
                             dropout=dropout,
                             head_n_hidden_layers=head_n_hidden_layers,
                             head_hidden_dim=head_hidden_dim,
                             final_nonlin=final_nonlin)

    return AttnMeanAndVarPoolMIL(encoder_dim=encoder_dim,
                                 encoder=instance_encoder,
                                 head=head,
                                 n_attn_latent=attn_latent_dim,
                                 gated=True,
                                 separate_attn=bool(separate_attn),
                                 n_var_pools=n_var_pools,
                                 act_func=var_act_func,
                                 dropout=dropout)


def get_model_summil(n_bag_feats,
                     out_dim,
                     dropout,
                     final_nonlin='relu',
                     head_n_hidden_layers=1,
                     attn_latent_dim=256,
                     head_hidden_dim=256):
    """
    Create Sum MIL model (simple baseline)
    """

    encoder_dim = 512

    instance_encoder = [nn.Linear(n_bag_feats, encoder_dim), nn.ReLU()]
    if dropout:
        instance_encoder.append(nn.Dropout(p=0.25))
    instance_encoder = nn.Sequential(*instance_encoder)

    # Head transfomrs the mean pool
    head = _get_head_network(encoder_dim=encoder_dim,
                             out_dim=out_dim,
                             dropout=dropout,
                             head_n_hidden_layers=head_n_hidden_layers,
                             head_hidden_dim=head_hidden_dim,
                             final_nonlin=final_nonlin)

    return SumMIL(encoder_dim=encoder_dim,
                  encoder=instance_encoder,
                  head=head,
                  n_attn_latent=attn_latent_dim,
                  dropout=dropout)


def get_model_summil_varpool(n_bag_feats,
                             out_dim,
                             n_var_pools,
                             var_act_func,
                             dropout,
                             final_nonlin='relu',
                             head_n_hidden_layers=1,
                             attn_latent_dim=256,
                             head_hidden_dim=256):
    """
    Create Sum MIL model with varpool head
    """

    encoder_dim = 512

    instance_encoder = [nn.Linear(n_bag_feats, encoder_dim), nn.ReLU()]
    if dropout:
        instance_encoder.append(nn.Dropout(p=0.25))
    instance_encoder = nn.Sequential(*instance_encoder)

    # Head transfomrs the mean pool
    head = _get_head_network(encoder_dim=encoder_dim + n_var_pools,
                             out_dim=out_dim,
                             dropout=dropout,
                             head_n_hidden_layers=head_n_hidden_layers,
                             head_hidden_dim=head_hidden_dim,
                             final_nonlin=final_nonlin)

    return SumVarPoolMIL(encoder_dim=encoder_dim,
                         encoder=instance_encoder,
                         head=head,
                         n_attn_latent=attn_latent_dim,
                         n_var_pools=n_var_pools,
                         act_func=var_act_func,
                         dropout=dropout)


def get_model_patchGCN(n_bag_feats,
                       out_dim,
                       dropout,
                       final_nonlin='relu'):
    """
    Load saved model from Patch-GCN paper for evaluation
    """

    # Default features
    hidden_dim = 128
    num_layers = 4
    resample = 0
    edge_agg = 'spatial'
    head_n_hidden_layers = 1
    n_classes = 4

    # Head transfomrs the mean pool
    head = _get_head_network(encoder_dim=hidden_dim * 4,
                             out_dim=out_dim,
                             dropout=dropout,
                             head_n_hidden_layers=head_n_hidden_layers,
                             head_hidden_dim=hidden_dim * 4,
                             final_nonlin=final_nonlin)

    return PatchGCN(num_features=n_bag_feats,
                    num_layers=num_layers,
                    edge_agg=edge_agg,
                    resample=resample,
                    dropout=dropout,
                    n_classes=n_classes,
                    head=head)


def get_model_patchGCN_varpool(n_bag_feats,
                               out_dim,
                               n_var_pools,
                               var_act_func,
                               dropout,
                               final_nonlin='relu'):

    # Default features
    hidden_dim = 128
    num_layers = 4
    resample = 0
    edge_agg = 'spatial'
    head_n_hidden_layers = 1
    n_classes = 4

    # Head transfomrs the mean pool
    head = _get_head_network(encoder_dim=hidden_dim * 4 + n_var_pools,
                             out_dim=out_dim,
                             dropout=dropout,
                             head_n_hidden_layers=head_n_hidden_layers,
                             head_hidden_dim=hidden_dim * 4,
                             final_nonlin=final_nonlin)

    return PatchGCN_varpool(num_features=n_bag_feats,
                            num_layers=num_layers,
                            edge_agg=edge_agg,
                            resample=resample,
                            dropout=dropout,
                            n_var_pools=n_var_pools,
                            act_func=var_act_func,
                            n_classes=n_classes,
                            head=head)


def get_model_MIL_Graph_FC(n_bag_feats,
                           out_dim,
                           dropout,
                           final_nonlin='relu'):
    """
    Load saved model from Patch-GCN paper for evaluation
    """

    # Default features
    hidden_dim = 128
    resample = 0
    # edge_agg = 'spatial'
    head_n_hidden_layers = 1
    n_classes = 4

    # Head transfomrs the mean pool
    head = _get_head_network(encoder_dim=hidden_dim,
                             out_dim=out_dim,
                             dropout=dropout,
                             head_n_hidden_layers=head_n_hidden_layers,
                             head_hidden_dim=hidden_dim,
                             final_nonlin=final_nonlin)

    return MIL_Graph_FC(num_features=n_bag_feats,
                        resample=resample,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        n_classes=n_classes,
                        head=head)


def get_model_MIL_Graph_FC_varpool(n_bag_feats,
                                   out_dim,
                                   n_var_pools,
                                   var_act_func,
                                   dropout,
                                   final_nonlin='relu'):
    """
    Load saved model from Patch-GCN paper for evaluation
    """

    # Default features
    hidden_dim = 128
    resample = 0
    # edge_agg = 'spatial'
    head_n_hidden_layers = 1
    n_classes = 4

    # Head transfomrs the mean pool
    head = _get_head_network(encoder_dim=hidden_dim + n_var_pools,
                             out_dim=out_dim,
                             dropout=dropout,
                             head_n_hidden_layers=head_n_hidden_layers,
                             head_hidden_dim=hidden_dim,
                             final_nonlin=final_nonlin)

    return MIL_Graph_FC_varpool(num_features=n_bag_feats,
                                resample=resample,
                                hidden_dim=hidden_dim,
                                dropout=dropout,
                                n_var_pools=n_var_pools,
                                act_func=var_act_func,
                                n_classes=n_classes,
                                head=head)


def _get_final_nonlin(final_nonlin):
    """
    Simple function to return corresponding nonlinear activation

    Parameters
    -----------
    final_nonlin: str
        Must be one of ['relu', 'tanh', 'identify']

    Output
    ------
    act_func: nn.Module
    """
    assert final_nonlin in ['relu', 'tanh', 'identity']

    if final_nonlin == 'relu':
        return nn.ReLU()
    elif final_nonlin == 'tanh':
        return nn.Tanh()
    elif final_nonlin == 'identity':
        return nn.Identity()


def _get_head_network(encoder_dim, out_dim, dropout, head_n_hidden_layers=1,
                      head_hidden_dim=256, final_nonlin='relu',
                      as_list=False):
    """

    Parameters
    -----------
    encoder_dim: int

    out_dim: int

    dropout: bool

    head_n_hidden_layers: int

    head_hidden_dim: int

    final_nonlin: str

    Output
    ------
    nn.Module
    """
    # head transforms the mean pooled features
    head = []
    if head_n_hidden_layers == 0:
        if head_hidden_dim != encoder_dim:
            warn("If hidden head layers then we need "
                 "encoder_dim == head_hidden_dim")

            head_hidden_dim = encoder_dim

        # assert head_hidden_dim == encoder_dim,\
        #     "if no hidden head layers then should have "\
        #     "encoder_dim == head_hidden_dim, but got "\
        #     "{}, {}".format(encoder_dim, head_hidden_dim)

    elif head_n_hidden_layers >= 1:

        # all but the last hidden layer gets a relu
        for L in range(head_n_hidden_layers):

            if L == 0:  # first layer
                in_features = encoder_dim
                out_features = head_hidden_dim

            else:  # any other layer
                in_features = head_hidden_dim
                out_features = head_hidden_dim

            if L == head_n_hidden_layers - 1:  # very last layer
                # last hidden layer gets possibly different non-linearity
                act_func = _get_final_nonlin(final_nonlin)
            else:
                act_func = nn.ReLU()

            head.extend([nn.Linear(in_features, out_features), act_func])
            if dropout:
                head.append(nn.Dropout(p=0.25))

    head.append(nn.Linear(head_hidden_dim, out_dim))

    if not as_list:
        head = nn.Sequential(*head)

    return head
