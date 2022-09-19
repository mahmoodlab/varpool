import torch.nn as nn
import torch

from itertools import combinations


class SurvRankingLoss(nn.Module):
    """
    Implements the surivival ranking loss which approximates the negaive c-index; see Section 3.2 of (Luck et al, 2018) -- but be careful of the typo in their c-index formula.

    The c-index for risk scores z_1, ..., z_n is given by

    c_index = sum_{(a, b) are comparable} 1(z_a > z_b)

    where (a, b) are comparable if and only if a's event is observed and a has a strictly lower survival time than b. This ignores ties.

    We replace the indicator with a continous approximation

    1(z_a - z_b > 0 ) ~= phi(z_a - z_b)

    e.g. where phi(r) is a Relu or sigmoid function.

    The loss function we want to minimize is then

    - sum_{(a, b) are comparable} phi(z_a - z_b)

    where z_a, z_b are the risk scores output by the network.

    Parameters
    ----------
    phi: str
        Which indicator approximation to use. Must be one of ['relu', 'sigmoid'].

    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']

    References
    ----------
    Luck, M., Sylvain, T., Cohen, J.P., Cardinal, H., Lodi, A. and Bengio, Y., 2018. Learning to rank for censored survival data. arXiv preprint arXiv:1806.01984.
    """

    def __init__(self, phi='sigmoid', reduction='mean'):
        super().__init__()

        assert phi in ['sigmoid', 'relu']
        assert reduction in ['mean', 'sum']
        self.phi = phi
        self.reduction = reduction

    def forward(self, z, c_t):
        """
        Parameters
        ----------
        z: (batch_size, 1)
            The predicted risk scores.

        c_t: (batch_size, 2)
            first element: censorship
            second element: survival time
        """
        batch_size = z.shape[0]

        if batch_size == 1:
            raise NotImplementedError("Batch size must be at least 2")

        censorship, times = c_t[:, 0], c_t[:, 1]
        events = 1 - censorship

        ##############################
        # determine comparable pairs #
        ##############################
        Z_more_risky = []
        Z_less_risky = []
        for (idx_a, idx_b) in combinations(range(batch_size), 2):
            time_a, event_a = times[idx_a], events[idx_a]
            time_b, event_b = times[idx_b], events[idx_b]

            if time_a < time_b and event_a:
                # a and b are comparable, a is more risky
                Z_more_risky.append(z[idx_a])
                Z_less_risky.append(z[idx_b])

            elif time_b < time_a and event_b:
                # a and b are comparable, b is more risky
                Z_more_risky.append(z[idx_b])
                Z_less_risky.append(z[idx_a])

        # if there are no comparable pairs then just return zero
        if len(Z_less_risky) == 0:
            # TODO: perhaps return None?
            return torch.zeros(1, requires_grad=True)

        Z_more_risky = torch.stack(Z_more_risky)
        Z_less_risky = torch.stack(Z_less_risky)

        # compute approximate c indices
        r = Z_more_risky - Z_less_risky
        if self.phi == 'sigmoid':
            approx_c_indices = torch.sigmoid(r)

        elif self.phi == 'relu':
            approx_c_indices = torch.relu(r)

        # negative mean/sum of c-indices
        if self.reduction == 'mean':
            return - approx_c_indices.mean()
        if self.reduction == 'sum':
            return - approx_c_indices.sum()
