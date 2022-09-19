import os
import yaml
from numbers import Number
from time import time
from datetime import datetime

from var_pool.file_utils import join_and_make


def run_train(tune_params, tune_idx, device,
              script_fpath, train_dir,
              fixed_params, fixed_flags,
              skip_if_results_exist=False):
    """
    Calls train.py for one tuning parameter setting and saves the tuning parameters settings to disk.

    Parameters
    ----------
    tune_idx: int
        Index of this tune setting.

    tune_params: dict
        The parameters that are tuned over and saved to disk. Any boolean values will be considered flags in the command string (and only called if they are True).

    device: None, int, list of int
        Which cuda devices(s) to use.

    script_fpath: str
        Path to the train.py script.

    train_dir: str
        Directory that containes the tune folders as tune_1, tune_2, ...

    fixed_params: dict
        The fixed parameters for each tune setting

    fixed_flags: list of str
        The flags that fixed for each tune setting.

    skip_if_results_exist: bool
        Dont run train.py if the corresponding results file already exists. Useful for picking up if tuning crashed half way through!
    """

    name = 'tune_{}'.format(tune_idx)
    tune_save_dir = join_and_make(train_dir, name)

    # Maybe skip this if results is already there
    results_fpath = os.path.join(tune_save_dir, 'results.yaml')
    if skip_if_results_exist and os.path.exists(results_fpath):
        print("Skipping tune_idx {} because the results file exists".
              format(tune_idx))
        return

    # Save the tunable parameters
    os.makedirs(tune_save_dir, exist_ok=True)
    tune_params_fpath = os.path.join(tune_save_dir, 'tune_params.yaml')
    with open(tune_params_fpath, 'w') as f:
        yaml.dump(tune_params, f)

    ##########################
    # setup parameter string #
    ##########################
    all_params = {**tune_params, **fixed_params}
    all_params['name'] = name
    all_flags = fixed_flags

    arg_str = ''
    for (k, v) in all_params.items():

        if isinstance(v, bool):
            # bools get turned into flags
            if v:
                arg_str += ' --{}'.format(k)
        else:
            arg_str += ' --{} {}'.format(k, v)

    for flag in all_flags:
        arg_str += ' --{}'.format(flag)

    ######################
    # setup cuda devices #
    ######################
    if device is None:
        cuda_prefix = ''
    elif isinstance(device, Number):
        cuda_prefix = 'CUDA_VISIBLE_DEVICES={}'.format(int(device))
    else:
        # list of devices
        cuda_prefix = 'CUDA_VISIBLE_DEVICES='
        cuda_prefix += ''.join([str(int(d)) for d in device])

    #################
    # setup command #
    #################
    command = cuda_prefix + ' python {}'.format(script_fpath) + arg_str

    current_time = datetime.now().strftime("%H:%M:%S")
    print("\n\n\n=====================================")
    print("Starting tune index {} at {} with params {}".
          format(tune_idx, current_time, tune_params))
    print(command)
    print("=====================================")
    start_time = time()
    os.system(command)

    runtime = time() - start_time
    print("=====================================")
    print("Finished running tune index {} after {:1.2f} seconds "
          "with parameters {}".
          format(tune_idx, runtime, tune_params))
    print("=====================================\n\n\n")
