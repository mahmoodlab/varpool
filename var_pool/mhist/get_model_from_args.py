from warnings import warn

from var_pool.mhist.get_model import (
    get_model_with_nn_layers,
    get_model_slim,
    get_model_varpool,
    get_model_summil,
    get_model_summil_varpool,
    get_model_patchGCN,
    get_model_patchGCN_varpool,
    get_model_MIL_Graph_FC,
    get_model_MIL_Graph_FC_varpool
)

# from var_pool.mhist.get_model_with_switch import get_model_varpool_with_switch


def get_model(args, n_bag_feats, out_dim):

    # load model object
    if args.arch == "amil_nn":
        return get_model_with_nn_layers(
            n_bag_feats=n_bag_feats,
            out_dim=out_dim,
            final_nonlin=args.final_nonlin,
            dropout=args.dropout,
            head_n_hidden_layers=args.head_n_hidden_layers,
            attn_latent_dim=args.attn_latent_dim,
            head_hidden_dim=args.head_hidden_dim,
        )

    elif args.arch == "amil_slim":
        return get_model_slim(
            n_bag_feats=n_bag_feats,
            out_dim=out_dim,
            final_nonlin=args.final_nonlin,
            dropout=args.dropout,
            attn_latent_dim=args.attn_latent_dim,
        )

    elif args.arch == "sum_mil":
        return get_model_summil(
            n_bag_feats=n_bag_feats,
            out_dim=out_dim,
            final_nonlin=args.final_nonlin,
            dropout=args.dropout,
            head_n_hidden_layers=args.head_n_hidden_layers,
            attn_latent_dim=args.attn_latent_dim,
            head_hidden_dim=args.head_hidden_dim,
        )

    elif args.arch == "sum_var_mil":
        if args.freeze_var_epochs is not None and args.freeze_var_epochs > 0:
            warn("Var freezing not implemented for sum_var_mil")

        return get_model_summil_varpool(
            n_bag_feats=n_bag_feats,
            out_dim=out_dim,
            n_var_pools=args.n_var_pools,
            var_act_func=args.var_act_func,
            dropout=args.dropout,
            head_n_hidden_layers=args.head_n_hidden_layers,
            attn_latent_dim=args.attn_latent_dim,
            head_hidden_dim=args.head_hidden_dim,
        )

    elif args.arch == "amil_var_nn":
        # Turn off switching for now
        return get_model_varpool(
            n_bag_feats=n_bag_feats,
            out_dim=out_dim,
            final_nonlin=args.final_nonlin,
            n_var_pools=args.n_var_pools,
            var_act_func=args.var_act_func,
            separate_attn=args.separate_attn,
            dropout=args.dropout,
            head_n_hidden_layers=args.head_n_hidden_layers,
            attn_latent_dim=args.attn_latent_dim,
            head_hidden_dim=args.head_hidden_dim,
        )
        # return get_model_varpool_with_switch(
        #     n_bag_feats=n_bag_feats,
        #     out_dim=out_dim,
        #     final_nonlin=args.final_nonlin,
        #     n_var_pools=args.n_var_pools,
        #     var_act_func=args.var_act_func,
        #     separate_attn=args.separate_attn,
        #     dropout=args.dropout,
        #     head_n_hidden_layers=args.head_n_hidden_layers,
        #     attn_latent_dim=args.attn_latent_dim,
        #     head_hidden_dim=args.head_hidden_dim,
        # )

    elif args.arch == "patchGCN":
        assert args.mode == "graph", "Patch GCN needs to be on graph mode"
        return get_model_patchGCN(
            n_bag_feats=n_bag_feats,
            out_dim=out_dim,
            final_nonlin=args.final_nonlin,
            dropout=args.dropout,
        )

    elif args.arch == "patchGCN_varpool":
        if args.freeze_var_epochs is not None and args.freeze_var_epochs > 0:
            warn("Var freezing not implemented for patchGCN_varpool")

        assert args.mode == "graph", "Patch GCN needs to be on graph mode"
        return get_model_patchGCN_varpool(
            n_bag_feats=n_bag_feats,
            out_dim=out_dim,
            n_var_pools=args.n_var_pools,
            var_act_func=args.var_act_func,
            final_nonlin=args.final_nonlin,
            dropout=args.dropout,
        )

    elif args.arch == 'amil_gcn':
        assert args.mode == "graph", "AMIL GCN needs to be on graph mode"
        return get_model_MIL_Graph_FC(
            n_bag_feats=n_bag_feats,
            out_dim=out_dim,
            final_nonlin=args.final_nonlin,
            dropout=args.dropout,
        )

    elif args.arch == 'amil_gcn_varpool':
        assert args.mode == "graph", "AMIL GCN needs to be on graph mode"
        return get_model_MIL_Graph_FC_varpool(
            n_bag_feats=n_bag_feats,
            out_dim=out_dim,
            n_var_pools=args.n_var_pools,
            var_act_func=args.var_act_func,
            final_nonlin=args.final_nonlin,
            dropout=args.dropout,
        )

    else:
        raise NotImplementedError("Not implemented for {}!".format(args.arch))
