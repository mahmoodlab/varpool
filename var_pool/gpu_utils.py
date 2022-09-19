import subprocess
import numpy as np
import torch
import time


def assign_free_gpus(max_gpus=3):
    """
    Identify least-utilized GPU and return the corresponding ID

    Parameters
    ----------
    max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                              Defaults to 3.
    """

    # Get the list of GPUs via nvidia-smi
    smi_query_result = subprocess.check_output('nvidia-smi -q -d Memory | grep -A4 GPU', shell=True)
    # Extract the usage information
    gpu_info = smi_query_result.decode('utf-8').split('\n')
    gpu_info = list(filter(lambda info: 'Used' in info, gpu_info))
    gpu_info = np.array([int(x.split(':')[1].replace('MiB', '').strip()) for x in gpu_info]) # Remove garbage

    # Delay by random amount to prevent choosing same gpus
    t_sleep = np.random.uniform(0, 30)
    print("Sleeping for ", t_sleep)
    time.sleep(t_sleep)

    # Tie breaking
    indices = np.where(gpu_info == gpu_info.min())[0]
    gpu_id = np.random.choice(indices)
    # if len(indices) == 1:
    #     gpu_id = gpu_info.min()
    # else:
    #     gpu_id = np.random.choice(np.arange(max_gpus, dtype=int))

    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    print("----------------------------------")
    print("Identified GPU with minimal usage ", np.where(gpu_info == gpu_info.min())[0])
    print("Training with device {}".format(device))
    print("----------------------------------")
    return device
