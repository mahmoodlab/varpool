"""
Dataset for graph neural network. Assumes the Data object has been already created
"""
import torch
from torch_geometric.data import Data


class GraphDataset:
    """
    Graph Dataset.

    Parameters
    ----------
    fpaths: list of str
        list of graph .pt files
    y: Dataframe
        Dataframe holding clinical information
    task: str
        rank_surv, cox_surv, or discr_surv
    arch: str
        Graph architecture to decide edge_spatial vs. edge_latent

    Outputs
    -------
    graph_list: list of torch_geometric.data.Data

    """
    def __init__(self, fpaths, y, task, arch):
        self.fpaths = fpaths
        self.y = y
        self.task = task
        self.arch = arch

    def __call__(self):
        graph_list = []
        fnames = ['-'.join(f.rsplit('/')[-1].split('-')[:3])
                  for f in self.fpaths]

        for idx, name in enumerate(fnames):
            try:
                df = self.y.loc[name]
            except KeyError:
                pass
            else:
                label = int(df['time_bin'])
                c = int(df['censorship'])
                t = float(df['survival_time'])

                if self.task == 'discr_surv':
                    y_out = torch.Tensor([label, c, t]).reshape(1, -1)
                else:
                    y_out = torch.Tensor([c, t]).reshape(1, -1)

                for slide_path in self.fpaths[name]:
                    temp = torch.load(slide_path)

                    if self.arch in ['amil_gcn', 'amil_gcn_varpool']:
                        graph_data = Data(edge_index=temp.edge_latent, x=temp.x, y=y_out)
                    else:
                        graph_data = Data(edge_index=temp.edge_index, x=temp.x, y=y_out)
                    graph_list.append(graph_data)

        return graph_list

# class GraphDataset:
#     """
#     Graph Dataset.
#
#     Parameters
#     ----------
#     data_dir: str
#         Directory where graph .pt files are located
#     y: Dataframe
#         Dataframe holding clinical information
#     task: str
#         rank_surv, cox_surv, or discr_surv
#
#     Outputs
#     -------
#     graph_list: list of torch_geometric.data.Data
#
#     """
#     def __init__(self, data_dir, y, task):
#         self.data_dir = data_dir
#         self.y = y
#         self.task = task
#
#     def __call__(self):
#         graph_list = []
#         flist = glob(os.path.join(self.data_dir, '*'))
#         fnames = ['-'.join(f.rsplit('/')[-1].split('-')[:3]) for f in flist]
#
#         for idx, f in enumerate(fnames):
#             try:
#                 df = self.y.loc[f]
#             except KeyError:
#                 pass
#             else:
#                 label = int(df['time_bin'])
#                 c = int(df['censorship'])
#                 t = float(df['survival_time'])
#
#                 if self.task == 'discr_surv':
#                     y_out = torch.Tensor([label, c, t]).reshape(1, -1)
#                 else:
#                     y_out = torch.Tensor([c, t]).reshape(1, -1)
#
#                 temp = torch.load(flist[idx])
#                 graph_data = Data(edge_index=temp.edge_index, x=temp.x, y=y_out)
#                 graph_list.append(graph_data)
#
#         return graph_list
