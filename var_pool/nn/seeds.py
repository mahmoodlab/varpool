import os
import numpy as np
import torch
import random


def set_seeds(device, seed=1):
    """
    Sets seeds to get reproducible experiments.

    Parametrs
    ---------
    device: torch.device
        The device we are using.

    seed: int
        The seed.
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
