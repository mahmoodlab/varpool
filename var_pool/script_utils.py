import os
import yaml
from pprint import pformat
import numpy as np

from var_pool.nn.utils import get_network_summary


def parse_mil_task_yaml(fpath):
    """
    Parameters
    ----------
    fpath: str
        Path to yaml file containging  task information.

    Output
    ------
    feat_dir, y_fpath, train_dir, task
    """
    assert os.path.exists(fpath), 'No file named {} found'.format(fpath)

    with open(fpath) as file:
        data = yaml.safe_load(file)

    return data['feats_dir'], data['y_fpath'], data['train_dir'], data['task']


def write_training_summary(fpath, task, y_split, args,
                           runtime, val_loss, val_metrics,
                           train_metrics, train_loss,
                           model, loss_func,
                           n_bag_feats, n_epoch_completed,
                           train_bag_size_summary=None,
                           epochs_with_records=None):
    """
    Writes a summary of training a neural network.

    Parameters
    ----------
    fpath: str
        Filepath for where to save the text file.

    Output
    ------
    summary: str
        The summary string that was saved to disk.
    """
    with open(fpath, 'w') as f:
        f.write("Final validation loss: {:1.3f}".format(val_loss))
        f.write("\nFinal validation metrics\n")
        f.write(pformat(val_metrics))

        f.write("\n\nFinal train loss: {:1.3f}".format(train_loss))
        f.write("\nFinal train metrics\n")
        f.write(pformat(train_metrics))

        if epochs_with_records is not None:
            epochs_with_records = np.array(epochs_with_records) + 1
            f.write("\nRecords set on epochs: {}\n".format(epochs_with_records))

        runtime_min = runtime / 60
        f.write("\n\nTraining model took {:1.2f} minutes "
                "({:1.2f} minutes per epoch)".
                format(runtime_min, runtime_min / n_epoch_completed))
        f.write("\nTraining completed {}/{} epochs".
                format(n_epoch_completed, args.n_epochs))

        f.write("\n\ntask = {}\n".format(task))
        for k, y in y_split.items():
            f.write("Number of {} samples = {}\n".format(k, len(y)))

        f.write("\nNumber of bag features {}".format(n_bag_feats))

        if train_bag_size_summary is not None:
            f.write("\nTraining bag size summary \n")
            f.write(pformat(train_bag_size_summary))

        f.write("\n\nargs=")
        f.write(str(args))

        f.write('\n\n')
        f.write(str(loss_func))
        f.write('\n\n')
        f.write(get_network_summary(model))

    # return the summary we just saved
    with open(fpath, 'r') as f:
        summary = f.read()
    return summary


def write_test_summary(fpath, task, split, eval_loss, eval_metrics, y_eval,
                       n_bag_feats, eval_bag_size_summary, loss_func, model):
    """
    Writes a summary of the test results.

    Parameters
    ----------
    fpath: str
        Filepath for where to save the text file.

    Output
    ------
    summary: str
        The summary string that was saved to disk.
    """
    with open(fpath, 'w') as f:
        f.write("Final {} loss: {:1.3f}".format(split, eval_loss))
        f.write("\nFinal {} metrics\n".format(split))
        f.write(pformat(eval_metrics))

        f.write("\n\ntask = {}\n".format(task))
        f.write("{} {} samples\n".format(len(y_eval), split))

        f.write("\nNumber of bag features {}".format(n_bag_feats))
        f.write("\n{} bag size summary \n")
        f.write(pformat(eval_bag_size_summary))

        f.write('\n\n')
        f.write(str(loss_func))
        f.write('\n\n')
        f.write(get_network_summary(model))

    # return the summary we just saved
    with open(fpath, 'r') as f:
        summary = f.read()
    return summary


def descr_stats(values):
    """
    Returns a dict of descriptive statistics (mean, min, max, etc) of an array of values.
    """
    values = np.array(values).reshape(-1)

    return {'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'num': len(values)
            }
