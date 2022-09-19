import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from time import time

from var_pool.file_utils import get_file_names


# TODO: I want to move this to another module for readability, but when
# I do that, class BagDataset(ResponseMixin, Dataset) throws
# a TypeError: metaclass conflict
# I am very confused
# maybe see https://stackoverflow.com/questions/22973902/metaclass-conflict-when-separating-classes-into-separate-files
class ResponseMixin:
    """
    Mixin for the response tasks

    TODO: more detail documentaiont

    Parameters
    ----------
    y: pd.Series or pd.DataFrame
        The reponse data. For classification and regression tasks y should be a pd.Series. For discrete survival tasks y should be a pd.DataFrame with columns ['time_bin', 'censorship', 'survival_time']..

    task: str
        The type of the task. Must be one of 'clf', 'discr_surv', 'cox_surv', 'rank_surv', 'reg'].
    """

    def _check_y(self, y, task):
        """
        Checks y and task are formatting correctly
        """
        if y is not None:

            assert task is not None, \
                "If y is provided then task must also be provided"

            avail_tasks = ['clf', 'discr_surv', 'cox_surv', 'rank_surv', 'reg']
            assert task in avail_tasks,\
                "task must be one of {}, not {}".\
                format(avail_tasks, task)

            if task == 'clf':
                assert isinstance(y, pd.Series),\
                    'y must be a pd.Series for classification tasks'

            elif task == 'reg':
                assert isinstance(y, pd.Series),\
                    'y must be a pd.Series for regression tasks'

            elif task == 'discr_surv':

                assert isinstance(y, pd.DataFrame) and \
                    'time_bin' in y.columns and \
                    'censorship' in y.columns and \
                    'survival_time' in y.columns and \
                    "['time_bin', 'censorship', 'survival_time'] "\
                    "for discrete survival tasks"

            elif task in ['rank_surv', 'cox_surv']:

                assert isinstance(y, pd.DataFrame) and \
                    'censorship' in y.columns and \
                    'survival_time' in y.columns,\
                    "y must be a pd.DataFrame and have columns "\
                    "['censorship', 'survival_time'] "\


    def _get_y(self, name):
        """
        Gets the respone value for a sample.

        Parameters
        ----------
        name:
            Name of the sample; should correspond to the index of y.

        Output
        ------
        y_out: torch.Tensor

        for classificaiton
        y_out: int
            Index of the class label.

        for classificaiton
        y_out: float
            The response.

        for discr_surv
        y_out: list, (3, )
            y_out[0]: int
                 Index of the time_bin class label.

            y_out[1]: int
                 The censorship status indicator.

            y_out[2]: float
                 The survival time.

        for rank_surv/cox_surv
        y_out: list, (2, )
            y_out[0]: int
                 The censorship status indicator.

            y_out[2]: float
                 The survival time.
        """
        if self.task == 'clf':
            y_out = int(self.y.loc[name])

        elif self.task == 'reg':
            y_out = float(self.y.loc[name])

        elif self.task == 'discr_surv':
            label = int(self.y.loc[name, 'time_bin'])
            c = int(self.y.loc[name, 'censorship'])
            t = float(self.y.loc[name, 'survival_time'])

            y_out = [label, c, t]

        elif self.task in ['cox_surv', 'rank_surv']:
            c = int(self.y.loc[name, 'censorship'])
            t = float(self.y.loc[name, 'survival_time'])

            y_out = [c, t]

        # format to tensor
        y_out = np.array(y_out)
        y_out = torch.from_numpy(y_out)
        return y_out


class BagDataset(ResponseMixin, Dataset):
    """
    Dataset for bags for multi instance learning problems. Assumes the bag features have been precomputed and saved to disk.

    Parameters
    ----------
    bag_fpaths: list of str or dict of lists
        The file paths to the bags of features. Each file should contain all instance features for one bag. Currently accepts ['.h5', '.csv', '.pt'] files. The bag names are the file names (without extension).

        If a dict is provided, each bag can have multiple files that are concatenated together. The dict keys should be the bag names and the entries should be a list of all the file paths.

        For HDF5 files, the instance features should be saved under the key 'features', but this can be changed by modifying the _hdf5_feature_key attrubute.

        For csv files, the first column should be the sample ids/index which will get dropped! All other columns are used as features.

    y: None, pd.DataFrame, pd.Series
        (Optional) The response data. The indices should correspond to file names. For classification tasks y should be a pd.Series. For discrete survival tasks y should be a pd.DataFrame with columns ['time_bin', 'censorship'].

    task: None, str
        (Optional) What kind of task the response is. Must be one of ['clf', 'reg', 'discr_surv']. This argmuent must be provided if y is provided.

    mode: str
        (Optional) What kind of mode the dataset is. Must be one of ['patch', 'graph']

    load_on_init: bool
        Whether or not to load all the bags into disk when initializing this object (faster, but more memory) or load them when __getitem__ is called (slower, less memory).

    device: None, torch.device
        (Optional) Put the data onto this device.

    Attributes
    ----------
    bag_names: list of str
        Names of each bag.

    name2fpaths: dict of str or lists
        A dict whose keys are the bag names and whose entries are either the filepath or list of filepaths for the corresponding features.

    y: None, pd.Series, pd.DataFrame
        The supervised output data.

    task: str
        The task

    bag_data_: dict
        If load_on_init=True, this contains the bag features.
    """
    def __init__(self, bag_fpaths, y=None, task=None,
                 load_on_init=False, mode='patch', device=None):
        assert len(bag_fpaths) > 0, 'No files input'
        assert isinstance(bag_fpaths, (list, dict))

        # make sure we have unique bag names
        if isinstance(bag_fpaths, dict):
            self.bag_names = list(bag_fpaths.keys())
            self.name2fpaths = bag_fpaths

        elif isinstance(bag_fpaths, list):
            self.bag_names = get_file_names(bag_fpaths)
            self.name2fpaths = {self.bag_names[i]: fpath
                                for i, fpath in enumerate(bag_fpaths)}

        assert len(self.bag_names) == len(set(self.bag_names)),\
            "Each bag must have a unique name"
        # self.bag_fpaths = bag_fpaths # TODO: should we save this?

        # format and load y data
        if y is not None:
            # subset y to only include bags we have data for
            # e.g. those in names
            y = y.loc[self.bag_names]
            self._check_y(y=y, task=task)
        self.y = y
        self.task = task
        self.mode = mode

        # maybe load bag data into memory
        if load_on_init:
            start_time = time()
            self._load_bag_data_into_memory(device=device)
            print("Loading bag features into memory took {:1.2f} seconds".
                  format(time - start_time))

        self.load_on_init = load_on_init
        self.device = device

    def _load_bag_data_into_memory(self, device=None):
        """
        Loads all the bag data into memory.
        """
        self.bag_data_ = {}
        for idx, bag_name in enumerate(self.bag_names):
            bag = self.load_bag_features_from_disk(bag_name)

            if device is not None:
                bag = bag.to(device)

            self.bag_data_[bag_name] = bag

    def __len__(self):
        return len(self.bag_names)

    def __getitem__(self, idx):
        """
        Output
        ------
        bag_features or bag_features, y_out

        bag_features: tuple
            bag_features[0]: as above
            bag_features[1]: int
                The non-pad size indicating how many instances in the bag are not padding.

        for classificaiton tasks
        y_out: int
            Index of the class label.

        for discrete survival tasks
        y_out: list, (3, )
            y_out[0]: int
                 Index of the time_bin class label.

            y_out[1]: int
                 The censorship status indicator.

            y_out[2]: float
                 The survival time.

        for survival tasks with cox/rank loss
        y_out: list, (2, )
            y_out[0]: int
                 The censorship status indicator.

            y_out[1]: float
                 The survival time.
        """
        bag_name = self.bag_names[idx]

        # load bag features
        if self.load_on_init:
            bag = self.bag_data_[bag_name]
        else:
            bag = self.load_bag_features_from_disk(bag_name)

        # move to device
        if self.device is not None:
            bag = bag.to(self.device)

        #  maybe get response
        if self.y is None:
            return bag
        else:
            # file name is the index for y
            y_out = self._get_y(name=bag_name)

            # move to device
            if self.device is not None:
                y_out = y_out.to(self.device)

            return bag, y_out

    def load_bag_features_from_disk(self, bag_name):
        """
        Loads the bag features for this bag. If this bag has multiple file paths we will concatenate them.

        Parameters
        ----------
        bag_name: str
            The filepath to the features.

        Output
        ------
        bag_features: torch.Tensor, (n_instances, n_bag_feats)
            The features for this file.
        """

        fpaths = self.name2fpaths[bag_name]
        if isinstance(fpaths, str):
            # Just one file for this bag!
            return self.load_from_disk(fpaths)

        # load and concatenate all files for this bag
        bag_feats = []
        for fpath in fpaths:
            bag_feats.append(self.load_from_disk(fpath))

        if self.mode == 'patch':
            if len(bag_feats) == 1:
                return bag_feats[0]
            else:
                return torch.vstack(bag_feats)

        elif self.mode == 'graph':
            ## Working on a hack
            # bag_feats = GraphDatasets.from_data_list(bag_feats, update_cat_dims={'edge_latent': 1})
            # return bag_feats
            return bag_feats[0]

    def load_from_disk(cls, fpath):
        """
        Loads the bag features from a file into memory a torch.Tensor.

        Parameters
        ----------
        bag_name: str
            The filepath to the features.

        Output
        ------
        features: torch.Tensor, (n_instances, n_bag_feats)
            The features for this file.
        """
        if fpath.endswith('.h5'):
            with h5py.File(fpath, 'r') as hdf5_file:
                feats = cls._load_from_h5(hdf5_file)
                feats = np.array(feats)
                return torch.from_numpy(feats)

        elif fpath.endswith('.csv'):
            feats = pd.read_csv(fpath, index_col=0)
            return torch.from_numpy(feats.values)

        elif fpath.endswith('.pt'):
            feats = torch.load(fpath)
            return feats

        else:
            raise NotImplementedError("file type not yet supported for {}".
                                      format(fpath))

    def _load_from_h5(cls, hdf5_file):
        """
        Loads the instance features from an open hdf5 file. Assumes the instance features are stored under the key 'features'. For HDF5 files formatted differently users should create a subclass that overrites this method.
        """
        return hdf5_file['features']

    def get_bag_summary(self):
        """
        Returns a summary of each bag in the dataset.

        Output
        ------
        n_instances, n_features

        n_instances: list of ints
            Number of instances in each bag.

        n_features: int
            Number of instance features.
        """

        shapes = []

        if self.load_on_init:
            # we have already loaded the bag feats
            shapes = [self.bag_data_[bag_name].shape
                      for bag_name in self.bag_names]

        else:
            for bag_name in self.bag_names:

                # file path(s) for this bag
                fpaths = self.name2fpaths[bag_name]
                if isinstance(fpaths, str):
                    fpaths = [fpaths]  # force to be a list

                # get shape for each file for this bag
                n_instances = 0
                for fpath in fpaths:

                    if fpath.endswith('.h5'):
                        # for HDF5 we can avoid loading the file into memory
                        with h5py.File(fpath, 'r') as hdf5_file:
                            # this does not load into numpy!
                            this_shape = self._load_from_h5(hdf5_file).shape

                    else:
                        this_shape = self.load_from_disk(fpath).shape

                    n_instances += this_shape[0]
                    n_feats = this_shape[1]

                shapes.append((this_shape, n_feats))

        # format
        n_instances, n_feats = zip(*shapes)
        n_instances = list(n_instances)
        n_feats = n_feats[0]

        return n_instances, n_feats
