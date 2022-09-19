import torch.nn as nn

from var_pool.nn.arch.VarPool_switch import AttnMeanAndVarPoolMIL_with_switch as AMVPool_with_switch

from var_pool.mhist.get_model import _get_head_network


def get_switch_parts(n_bag_feats, encoder_dim, out_dim, n_var_pools,
                     head_hidden_dim, head_n_hidden_layers,
                     final_nonlin, dropout):

    ###############################
    # First layer of head network #
    ###############################
    if head_n_hidden_layers == 0:
        head_hidden_dim = out_dim

    mean_neck = nn.Linear(encoder_dim, head_hidden_dim)
    var_neck = nn.Linear(n_var_pools, head_hidden_dim)

    ########################
    # Rest of head network #
    ########################

    if head_n_hidden_layers >= 1:

        # head transforms the concatenation of the mean pool and variance pool
        head = _get_head_network(encoder_dim=head_hidden_dim,
                                 out_dim=out_dim,
                                 dropout=dropout,
                                 head_n_hidden_layers=head_n_hidden_layers - 1,
                                 head_hidden_dim=head_hidden_dim,
                                 final_nonlin=final_nonlin,
                                 as_list=True)

        prefix = [nn.ReLU()]
        if dropout:
            prefix.append(nn.Dropout(0.25))
        head = prefix + head

        head = nn.Sequential(*head)
    else:
        head = None

    return head, mean_neck, var_neck


def get_model_varpool_with_switch(n_bag_feats,
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

    head, mean_neck, var_neck = \
        get_switch_parts(n_bag_feats=n_bag_feats,
                         encoder_dim=encoder_dim,
                         out_dim=out_dim,
                         n_var_pools=n_var_pools,
                         head_hidden_dim=head_hidden_dim,
                         head_n_hidden_layers=head_n_hidden_layers,
                         final_nonlin=final_nonlin,
                         dropout=dropout)

    return AMVPool_with_switch(encoder_dim=encoder_dim,
                               encoder=instance_encoder,
                               mean_neck=mean_neck,
                               var_neck=var_neck,
                               head=head,
                               n_attn_latent=attn_latent_dim,
                               gated=True,
                               separate_attn=bool(separate_attn),
                               n_var_pools=n_var_pools,
                               act_func=var_act_func,
                               dropout=dropout)
