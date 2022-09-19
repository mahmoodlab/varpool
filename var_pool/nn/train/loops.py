import torch
from tqdm import tqdm

from torch.nn.modules.loss import _WeightedLoss
import torch_geometric

from var_pool.nn.stream_evaler import BaseStreamEvaler
from var_pool.nn.train.GradAccum import GradAccum


# TODO: think through grad accum and nn.DataParallel
# TODO: perhaps add override_sum_to_mean_reduction option to force (unweighted) mean reduction of a loss function e.g. see https://github.com/pytorch/pytorch/issues/72047#issuecomment-1027002874
def train_loop(model, loss_func, loader, optimizer,
               grad_accum=None,
               evaler=None, device=None, tqdm_desc='training batches',
               mode='patch'):
    """
    Runs a training loop.

    Parameters
    ----------
    model: nn.Module
        The model.

    loss_func: nn.Module
        The loss function.

    loader: DataLoader
        The training data loader.

    optimizer: Optimizer
        The optimizer.

    grad_accum: None, int
        (Optional) Number of gradient accumulation steps. This currently address the issue with gradient accumulation + weighted losses for batch_size > 1 (see https://github.com/pytorch/pytorch/issues/72047) but manually performing an unweighted mean reduction instead of the current weighted mean reduction.

    evaler: None, var_pool.nn.stream_evaler.BaseStreamEvaler
        (Optional) An object that computes supervised learning evaulation metrics e.g. classification error. This must be a subclass of BaseStreamEvaler() that handles computing the predictions in batches.

    device: None, str
        (Optional) Overwrite the default device; otherwise the device is selected in the standard way.

    tqdm_desc: None, str
        The descr argument for the tqdm progress bar. Setting this to None will disable the tqdm progress bar.

    mode: str
        'patch' or 'graph'

    Output
    ------
    epoch_loss, metrics

    epoch_loss: float
        The average loss for this epoch.

    metrics: dict, None
        Additional evaulation metrics computed by the evaler.
    """

    # setup pytorch stuff
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    # make sure to reset the tracked evaulation data
    if evaler is not None:
        assert isinstance(evaler, BaseStreamEvaler)
        evaler.reset_tracking()

    ####################################
    # Override weighted mean reduction #
    ####################################
    # pytorch does a weighted mean reduction for batchs by default which seems
    # like strange default behavior. We manually override this behavior
    # and do an unweighted reduction for batches
    force_unweighted_mean_reduction = True  # TODO: perhaps make this an option

    # Note this make the gradient accumulation adjustment below work correctly
    # e.g. see https://github.com/pytorch/pytorch/issues/72047

    if isinstance(loss_func, _WeightedLoss) and \
            loss_func.weight is not None and\
            loss_func.reduction == 'mean' and \
            loader.batch_size > 1 and \
            force_unweighted_mean_reduction:

        override_mean_reductuction = True
        loss_func.reduction = 'sum'

    else:
        override_mean_reductuction = False

    ################################
    # handle gradient accumulation #
    ################################
    grad_accum = None if (grad_accum is not None or grad_accum <= 1)\
        else grad_accum

    if grad_accum is not None and loss_func.reduction == 'mean':
        # helper to adjust the loss divisor for gradient accumulation
        # when we average over the sample losses
        GA_helper = GradAccum(loader=loader, grad_accum=grad_accum)
    else:
        # if we are not doing mean reduction then we dont need to adjust
        # the loss divisor below
        GA_helper = None

    epoch_loss = 0.
    model.zero_grad()
    update_params = True  # used by grad accum
    for batch_idx, batch in enumerate(tqdm(loader,
                                            desc=tqdm_desc,
                                            disable=tqdm_desc is None,
                                            position=1)):

        if mode == 'patch':
            data, y_true = batch
        elif mode == 'graph':
            data = batch
            y_true = batch.y

        # move data to device and get forward pass
        data = safe_list_to(data=data, device=device)
        y_true = y_true.to(device)
        z = model(data)
        loss = loss_func(z, y_true)

        if override_mean_reductuction:
            # manually compute loss average since we changed
            # reduction to sum
            batch_size = y_true.shape[0]
            loss = loss / batch_size

        # Handle gradient accumulation
        if grad_accum is not None and GA_helper is not None:
            # adjust divisor for gradient accumulation
            loss_div, update_params = GA_helper.get_loss_div(batch_idx)
            loss = loss / loss_div

        loss.backward()

        if grad_accum is None or update_params:
            optimizer.step()
            optimizer.zero_grad()

        # maybe track metrics
        epoch_loss += loss.item()
        if evaler is not None:
            evaler.log(z=z, y_true=y_true)

    # calculate loss and error for epoch
    epoch_loss /= len(loader)
    if evaler is not None:
        metrics = evaler.get_metrics()
    else:
        metrics = None

    if override_mean_reductuction:
        # put reduction back to mean as it originally was since we modified
        # it above
        loss_func.reduction = 'mean'

    return epoch_loss, metrics


def eval_loop(model, loader, loss_func=None,
              evaler=None, device=None,
              tqdm_desc='evaulation batches',
              mode='patch'):
    """
    Runs an evaluation loop.

    Parameters
    ----------
    model: nn.Module
        The model.

    loader: DataLoader
        The training data loader

    loss_func: None, nn.Module
        (Optional) The loss function if we want to evaulate it.

    evaler: None, var_pool.nn.stream_evaler.BaseStreamEvaler
        (Optional) An object that computes supervised learning evaulation metrics e.g. classification error. This must be a subclass of BaseStreamEvaler() that handles computing the predictions in batches.

    device: None, str
        (Optional) Overwrite the default device; otherwise the device is selected in the standard way.

    tqdm_desc: None, str
        The descr argument for the tqdm progress bar. Setting this to None will disable the tqdm progress bar.

    mode: str
        'patch' or 'graph'

    Output
    ------
    epch_loss, metrics

    epoch_loss: float
        The average loss for this epoch.

    metrics: dict, None
        Additional evaulation metrics computed by the evaler.
    """
    assert loss_func is not None or evaler is not None,\
        "Provided at least one of loss_func or evaler"
    # TODO: should this be an error?
    # if loader.batch_size > 1 and loss_func.reduction != 'sum':
    #     raise RuntimeError("For loss functions with mean reduction "
    #                        "the loader's batch_size should be 1 to "
    #                        "ensure we properly compute the loss.")

    # setup pytorch stuff
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # make sure to reset the tracked evaulation data
    if evaler is not None:
        assert isinstance(evaler, BaseStreamEvaler)
        evaler.reset_tracking()

    epoch_loss = 0 if loss_func is not None else None

    with torch.no_grad():
        for batch in tqdm(loader,
                          desc=tqdm_desc, disable=tqdm_desc is None,
                          position=1):

            if mode == 'patch':
                data, y_true = batch
            elif mode == 'graph':
                data = batch
                y_true = batch.y

            # move data to device and get forward pass
            data = safe_list_to(data=data, device=device)
            y_true = y_true.to(device)
            z = model(data)

            if loss_func is not None:
                loss = loss_func(z, y_true)
                epoch_loss += loss.item()

            # maybe track metrics
            if evaler is not None:
                evaler.log(z=z, y_true=y_true)

    # calculate loss and error for epoch
    if loss_func is not None:
        epoch_loss /= len(loader)

    if evaler is not None:
        metrics = evaler.get_metrics()
    else:
        metrics = None

    return epoch_loss, metrics


def safe_list_to(data, device):
    """
    Moves data to device when data is either a torch.Tensor or a list/tuple/dict of tensors.
    e.g. [d.to(device) for d in data]

    Parameters
    ----------
    data: torch.tensor, tuple, list, dict
        The input data to put on a device.

    device: torch.device
        The device to move data do.


    Output
    ------
    data: torch.tensor, tuple, list, dict
        Data or each element of data on the device preserving the input structure e.g. if a list was provided then a list will be output.
    """

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple):
        return (d.to(device) for d in data)
    elif isinstance(data, list):
        return [d.to(device) for d in data]
    elif isinstance(data, dict):
        return {k: v.to(device) for (k, v) in data.keys()}
    elif isinstance(data, torch_geometric.data.Batch):
        return data.to(device)
    else:
        raise RuntimeError("data should be a Tensor, tuple, list or dict, but"
                           " not {}".format(type(data)))
