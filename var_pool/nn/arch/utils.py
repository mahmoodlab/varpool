import torch.nn as nn


def mlp_from_tuple(dims, act='relu'):
    """
    Creates a multi-layer perceptron.

    Parameters
    ----------
    dims: list of ints
        The dimensions of the layers including the input and output layer i.e. there are len(dims) - 1 total layers

    act: str
        Activation function. Must be one of ['relu'].

    Output
    ------
    net: nn.Module
        The MLP network.
    """

    net = []
    n_layers = len(dims) - 1
    for layer in range(n_layers):

        net.append(nn.Linear(dims[layer], dims[layer+1]))

        # add Relus after all but the last layer
        if layer < len(dims) - 1:
            if act == 'relu':
                a = nn.ReLU()
            else:
                raise NotImplementedError
            net.append(a)

    return nn.Sequential(*net)
