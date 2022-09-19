import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from var_pool.nn.train.GradAccum import GradAccum


def check_GradAccum(n_samples, batch_size, grad_accum):
    """
    Checks the GradAccum() object by checking that gradient accumulation is equivalent to using a new batch size of batch_size x grad_accum.

    Parameters
    ----------
    n_samples: int
        Number of smaples in the dataset.

    batch_size: int
        the batch size.

    grad_accum: int
        Number of gradient accumulation setp.

    Output
    ------
    test_passes: bool
        Whether or not the test passes.
    """

    np.random.seed(1)
    torch.manual_seed(0)
    n_features = 5

    # n_samples = 4
    # batch_size = 4
    # grad_accum = 2

    # n_samples = 10
    # batch_size = 3
    # grad_accum = 2

    # Setup data set
    X = np.random.normal(size=(n_samples, n_features))
    y = np.random.normal(size=n_samples)
    model = LinearRegression(n_features)
    dataset = RegressionDataset(X, y)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # gradient accumulation with batch size B should be equivalent to using
    # batch size = B x grad_accum
    loader_BxG = DataLoader(dataset, batch_size=batch_size * grad_accum,
                            shuffle=False)

    # run loops
    ga = epoch_with_grad_accum(model=model, loader=loader,
                               grad_accum=grad_accum)
    BxG = epoch_manual(model=model, loader_BxG=loader_BxG)

    # check results
    return all([torch.allclose(a, b) for (a, b) in zip(ga, BxG)])
    # test_passes = True
    # for a, b in zip(ga, BxG):
    #     # print(a - b)
    #     # assert torch.allclose(a, b)
    #     if not torch.allclose(a, b):
    #         test_passes = False


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y).reshape(-1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


class LinearRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.coef = nn.Linear(n_features, 1)

    def forward(self, input):
        return self.coef(input)


def initialize_to_zero(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.zero_()
            m.bias.data.zero_()


def epoch_manual(model, loader_BxG, verbose=False):
    """
    Run one epoch with loader that uses batch size n_batches x grad accum.

    Output
    ------
    model_params: list of tensors
        The model parameters after one epoch.
    """

    # setup loss/optimizer
    loss_func = nn.MSELoss()
    initialize_to_zero(model)  # always init at zero
    optimizer = optim.SGD(params=model.parameters(),
                          lr=1, momentum=0, weight_decay=0)

    if verbose:
        print('initial parameters', list(model.parameters()), '\n')

    model.zero_grad()
    for batch_idx, (x, y_true) in enumerate(loader_BxG):

        x = x.float()
        y_true = y_true.unsqueeze(1)

        y_pred = model(x.float())
        loss = loss_func(y_true.float(), y_pred)
        loss.backward()

        if verbose:
            print('grad batch_idx={}'.format(batch_idx),
                  [p.grad for p in model.parameters()])

        optimizer.step()
        optimizer.zero_grad()

    return [p.data for p in model.parameters()]


def epoch_with_grad_accum(model, loader, grad_accum, verbose=False):
    """
    Run one epoch with gradient accumulation

    Output
    ------
    model_params: list of tensors
        The model parameters after one epoch.
    """

    GA_helper = GradAccum(loader=loader, grad_accum=grad_accum)

    # setup loss/optimizer
    loss_func = nn.MSELoss()
    initialize_to_zero(model)  # always init at zero
    optimizer = optim.SGD(params=model.parameters(),
                          lr=1, momentum=0, weight_decay=0)

    if verbose:
        print('initial parameters', list(model.parameters()), '\n')

    model.zero_grad()
    for batch_idx, (x, y_true) in enumerate(loader):
        x = x.float()
        y_true = y_true.unsqueeze(1)

        y_pred = model(x.float())
        loss = loss_func(y_true.float(), y_pred)

        # adjust loss divisor
        loss_div, update_params = GA_helper.get_loss_div(batch_idx)
        loss = loss / loss_div

        if verbose:
            print('loss_div batch_idx={}'.format(batch_idx), loss_div)

        loss.backward()

        # step after gradeint accumulation
        if update_params:
            if verbose:
                print('grad batch_idx={}'.format(batch_idx),
                      [p.grad for p in model.parameters()])

            optimizer.step()
            optimizer.zero_grad()

    if verbose:
        print()
        print(GA_helper.__dict__)

    return [p.data for p in model.parameters()]
