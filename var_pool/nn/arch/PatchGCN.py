"""
PatchGCN architecture from Richard's code
(TODO) Still work in the progress
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GENConv, DeepGCNLayer
# from torch_geometric.transforms.normalize_features import NormalizeFeatures

from var_pool.nn.arch.AttnMIL import GatendAttn
from var_pool.nn.arch.VarPool import VarPool


class PatchGCN(nn.Module):
    """
    Patch GCN, designed for NLL loss
    """
    def __init__(self, input_dim=2227, num_layers=3, edge_agg='spatial',
                 multires=False, resample=0, fusion=None, num_features=1024,
                 hidden_dim=128, linear_dim=64, use_edges=False, pool=False,
                 dropout=0.25, n_classes=4, head=None):

        super().__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)
            act = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = GatendAttn(n_in=hidden_dim*4,
                                              n_latent=hidden_dim*4,
                                              dropout=dropout)

        self.head = head

    def forward(self, batch):
        """
        Wrapper for forward pass of batch graph data

        Parameters
        ----------
        batch: torch_geometric.data.Batch object

        Output
        ------
        batch_output: (batch_size x out_dim)
        """
        data_list = batch.to_data_list()
        batch_out = []
        for data in data_list:
            out = self._forward(data)
            batch_out.append(out)

        batch_out = torch.cat(batch_out)
        # print("Batch ", batch_out.shape)
        return batch_out

    def _forward(self, data):

        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent

        edge_attr = None

        x = self.fc(data.x)
        x_ = x

        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)

        # h_path = x_
        # h_path = self.path_phi(h_path)
        #
        # print("=====================")
        # print("DATA ", data)
        # print("H_path ", h_path.shape)
        #
        # A_path, h_path = self.path_attention_head(h_path)
        # print("A_Path ", A_path.shape)
        # A_path = torch.transpose(A_path, 1, 0)
        # h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        # print("H_path 2", h_path.shape)
        # out = self.head(h_path)
        # print("Out ", out.shape)

        h_path = x_
        h_path = self.path_phi(h_path)

        attn, h_path = self.path_attention_head(h_path)
        attn = F.softmax(attn, dim=0)   # Normalized attention (n_instances, 1)

        weighted_avg_h = torch.mm(attn.transpose(1, 0), h_path)  # (1 x latent)
        out = self.head(weighted_avg_h)

        return out


class PatchGCN_varpool(nn.Module):
    """
    Patch GCN with variance pooling, designed for NLL loss
    """
    def __init__(self, input_dim=2227, num_layers=3, edge_agg='spatial',
                 multires=False, resample=0, fusion=None, num_features=1024,
                 hidden_dim=128, linear_dim=64, use_edges=False, pool=False,
                 dropout=0.25, n_classes=4, n_var_pools=100,
                 act_func='log', head=None):

        super().__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)
            act = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = GatendAttn(n_in=hidden_dim*4,
                                              n_latent=hidden_dim*4,
                                              dropout=dropout)

        self.head = head

        # Variance pooling
        self.var_pool = VarPool(encoder_dim=hidden_dim*4,
                                n_var_pools=n_var_pools,
                                act_func=act_func)

    def forward(self, batch):
        """
        batch: torch_geometric.data.Batch object

        """
        data_list = batch.to_data_list()
        batch_out = []
        for data in data_list:
            out = self._forward(data)
            batch_out.append(out)

        batch_out = torch.cat(batch_out)
        # print("Batch ", batch_out.shape)
        return batch_out

    def _forward(self, data):

        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent

        edge_attr = None
        x = self.fc(data.x)
        x_ = x

        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)

        h_path = x_
        h_path = self.path_phi(h_path)

        attn, h_path = self.path_attention_head(h_path)
        attn = F.softmax(attn, dim=0)   # Normalized attention (n_instances, 1)

        # print("H_path ", h_path.shape)
        # print("Attn ", attn.shape)

        weighted_avg_h = torch.mm(attn.transpose(1, 0), h_path)  # (1 x latent)
        var_pooled_h = self.var_pool(h_path.unsqueeze(0), attn.unsqueeze(0))   # (1 x latent)

        # print("Var pool ", var_pooled_h.shape)

        merged_h = torch.cat((weighted_avg_h, var_pooled_h), dim=1)

        out = self.head(merged_h)

        return out


class MIL_Graph_FC(nn.Module):
    """
    MIL Graph FC
    """
    def __init__(self, input_dim=2227,
                 multires=False, resample=0, fusion=None, num_features=1024,
                 hidden_dim=128, linear_dim=64, use_edges=False, pool=False,
                 dropout=0.25, n_classes=4, head=None):

        super().__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.multires = multires
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample),
                                      nn.Linear(num_features, 128),
                                      nn.ReLU(),
                                      nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(num_features, 128),
                                      nn.ReLU(),
                                      nn.Dropout(0.25)])

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.path_attention_head = GatendAttn(n_in=hidden_dim,
                                              n_latent=hidden_dim,
                                              dropout=dropout)

        self.head = head

    def forward(self, batch):
        """
        Wrapper for forward pass of batch graph data

        Parameters
        ----------
        batch: torch_geometric.data.Batch object

        Output
        ------
        batch_output: (batch_size x out_dim)
        """
        data_list = batch.to_data_list()
        batch_out = []
        for data in data_list:
            out = self._forward(data)
            batch_out.append(out)

        batch_out = torch.cat(batch_out)
        # print("Batch ", batch_out.shape)
        return batch_out

    def _forward(self, data):

        x = data.x
        e_latent = data.edge_index
        # batch = data.batch
        edge_attr = None

        x = self.fc(x)
        x1 = F.relu(self.conv1(x, e_latent, edge_attr))
        x2 = F.relu(self.conv2(x1, e_latent, edge_attr))

        h_path = x2
        # h_path = self.path_phi(h_path)

        attn, h_path = self.path_attention_head(h_path)
        attn = F.softmax(attn, dim=0)   # Normalized attention (n_instances, 1)

        weighted_avg_h = torch.mm(attn.transpose(1, 0), h_path)  # (1 x latent)
        out = self.head(weighted_avg_h)

        return out


class MIL_Graph_FC_varpool(nn.Module):
    """
    MIL Graph FC
    """
    def __init__(self, input_dim=2227,
                 multires=False, resample=0, fusion=None, num_features=1024,
                 hidden_dim=128, linear_dim=64, use_edges=False, pool=False,
                 dropout=0.25, n_classes=4, n_var_pools=100,
                 act_func='log', head=None):

        super().__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.multires = multires
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample),
                                      nn.Linear(num_features, 128),
                                      nn.ReLU(),
                                      nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(num_features, 128),
                                      nn.ReLU(),
                                      nn.Dropout(0.25)])

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.path_attention_head = GatendAttn(n_in=hidden_dim,
                                              n_latent=hidden_dim,
                                              dropout=dropout)

        self.head = head

        # Variance pooling
        self.var_pool = VarPool(encoder_dim=hidden_dim,
                                n_var_pools=n_var_pools,
                                act_func=act_func)

    def forward(self, batch):
        """
        Wrapper for forward pass of batch graph data

        Parameters
        ----------
        batch: torch_geometric.data.Batch object

        Output
        ------
        batch_output: (batch_size x out_dim)
        """
        data_list = batch.to_data_list()
        batch_out = []
        for data in data_list:
            out = self._forward(data)
            batch_out.append(out)

        batch_out = torch.cat(batch_out)
        # print("Batch ", batch_out.shape)
        return batch_out

    def _forward(self, data):

        x = data.x
        e_latent = data.edge_index
        # batch = data.batch
        edge_attr = None

        x = self.fc(x)
        x1 = F.relu(self.conv1(x, e_latent, edge_attr))
        x2 = F.relu(self.conv2(x1, e_latent, edge_attr))

        h_path = x2
        # h_path = self.path_phi(h_path)

        attn, h_path = self.path_attention_head(h_path)
        attn = F.softmax(attn, dim=0)   # Normalized attention (n_instances, 1)

        weighted_avg_h = torch.mm(attn.transpose(1, 0), h_path)  # (1 x latent)
        var_pooled_h = self.var_pool(h_path.unsqueeze(0), attn.unsqueeze(0))   # (1 x latent)
        merged_h = torch.cat((weighted_avg_h, var_pooled_h), dim=1)

        out = self.head(merged_h)

        return out
