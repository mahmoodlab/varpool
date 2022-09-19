import torch.nn as nn
import torch


class GlobalPoolMil(nn.Module):
    """
    Global pool the bags (e.g. mean pool) the on linear layer.

    Parameters
    ----------
    in_feats: int
        Feature input dimension.

    out_dim: int
        Output dimension.

    pool: str
        Which pooling operation to apply to each bag feature.
    """
    def __init__(self, in_feats, out_dim, pool='mean'):
        super().__init__()

        self.head = nn.Linear(in_feats, out_dim)

        assert pool in ['mean']
        self.pool = pool

    def forward(self, bags):
        if self.pool == 'mean':
            feats = torch.mean(bags, axis=1)
        return self.head(feats)

