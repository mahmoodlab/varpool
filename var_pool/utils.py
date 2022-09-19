from collections import Counter
import numpy as np


def format_command_line_args(kws={}, flags=[]):
    """
    Parameters
    ----------
    kws: dict
        Arguments that look like --key value. If a list is provided it will be input as --key value_1 value_2, ...

    flags: list of str
        Arguments that look like --flag

    Output
    ------
    args: str
        The argument string
    """

    args = ''
    for key, value in kws.items():

        args += ' --{}'.format(key)

        if np.isscalar(value):
            args += ' {}'.format(value)
        else:
            for v in value:
                args += ' {}'.format(v)

    for flag in flags:
        args += ' --{}'.format(flag)

    return args


def get_counts_and_props(y, n_classes=None, class_names=None):
    """
    Gets the counts and proportions for each class from a vector of class index predictions.

    Parameters
    ----------
    y: array-like, (n_samples, )
        The predicted class indices.

    n_classes: None, int
        (Optional) The number of class labels; if not provided will try to guess by either len(class_names) (if provided) or  max(y) + 1.

    class_names: None, str, list of str
        (Optional) If provided will return the counts/props as a dict with the class names as keys. If set to 'default' then will name the classes class_0, class_1, ... If None, will return the counts/props as lists

    Output
    ------
    counts, props

    if class_names==None
    counts: array-like, (n_classes, )
        The counts for each class.

    props: array-like, (n_classes, )
        The proportions for each class.

    if class_names is provided then counts/props will be a dicts
    """
    y = np.array(y).reshape(-1)

    if n_classes is None:
        if class_names is not None and not isinstance(class_names, str):
            n_classes = len(class_names)

        else:
            # +1 bc of zero indexing!
            n_classes = max(y) + 1

    ########################
    # compute counts/props #
    ########################
    counts = np.zeros(n_classes)
    for idx, cnt in Counter(y).items():
        counts[idx] = cnt

    props = counts / len(y)

    #################
    # Format output #
    #################
    if class_names is None:
        return counts, props

    if isinstance(class_names, str) and class_names == 'default':
        class_names = ['class_' + str(i) for i in range(n_classes)]

    counts_dict = {}
    props_dict = {}
    for cl_idx, name in enumerate(class_names):

        counts_dict[name] = counts[cl_idx]
        props_dict[name] = props[cl_idx]

    return counts_dict, props_dict


def get_traceback(e):
    """
    Returns the traceback from any expection.
    Parameters
    ----------
    e: BaseException
        Any excpetion.
    Output
    ------
    str
    """
    # https://stackoverflow.com/questions/3702675/how-to-catch-and-print-the-full-exception-traceback-without-halting-exiting-the
    import traceback
    return ''.join(traceback.format_exception(None, e, e.__traceback__))
