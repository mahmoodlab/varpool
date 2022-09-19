import torch.nn as nn
import torch.optim as optim


def get_network_summary(net):
    """
    Prints a summary of neural network including the number of parameters.

    Output
    ----------
    summary: str
        A text summary of the network.
    """

    num_params = 0
    num_params_train = 0

    summary = str(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    summary += '\n\nTotal number of parameters: {}'.\
        format(num_params)
    summary += '\nTotal number of trainable parameters: {}'.\
        format(num_params_train)

    return summary


def initialize_weights(module):
    """
    Initializes the weights for a neural network.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def get_optim(model, algo='adam', lr=1e-4, weight_decay=1e-5):
    """
    Sets up the optimizer for the trainable parametres.

    Parameters
    ----------
    model:

    algo: str
        The optimization algorithm. Must be one of ['adam', 'sgd']

    lr: float
        The learning rate.

    weight_decay: None, float
        Weight decay (L2 penalty)

    Output
    ------
    optim: torch.optim.Optimizer
        The setup optimizer.
    """

    # pull out trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if algo == "adam":
        optimizer = optim.Adam(params=trainable_params, lr=lr,
                               weight_decay=weight_decay)

    elif algo == 'sgd':
        optimizer = optim.SGD(params=trainable_params, lr=lr,
                              momentum=0.9, weight_decay=weight_decay)

    else:
        raise NotImplementedError("{} not currently implemented".format(algo))

    return optimizer
