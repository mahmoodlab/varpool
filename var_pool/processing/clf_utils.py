import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


def dict_split_clf_df(y_df, label_col='label',
                      split_col='split', index_col='sample_id'):
    """
    Splits the y label data frame into train, validation, and test sets.

    Parameters
    ----------
    df: pd.DataFrame
        A data frame whose first three columns are  ['sample_id', 'label', 'split']
        'sample_id' is the identifier for each sample
        'label' is the class label
        'split' should be one of ['train', 'val', 'test'] indicating which hold out set the sample is in

    label_col: str
        Name of the column containing the y label.

    spilt_col: str
        Name of the column containing the train/test/val splits.

    index_col: str
        Name of the column containin the sample identifies; this will be used to index the pd.Series in y_split.

    Output
    ------
    y_split, label_enc

    y_split: dict of pd.Series
        The keys include ['train', 'val', 'test']. The series include the class labels converted to indices. The pd.Series are indexed by the index_col.

    label_enc: sklearn.preprocessing.LabelEncoder
        The label encoder use to convert the sample categories to indices.
    """

    # pull out the important columns and copy.
    cols = [index_col, label_col, split_col]
    y_df = y_df[cols].copy()

    # tranform class names to indices
    label_enc = LabelEncoder()
    y_idxs = label_enc.fit_transform(y_df[label_col].values)
    y_df[label_col] = y_idxs

    # split into train/val/test sets
    y_split = {}
    for kind, df in y_df.groupby(split_col):
        assert kind in ['train', 'val', 'test']

        # make pandas series for each split
        y_split[kind] = pd.Series(df[label_col].values,
                                  name='label',
                                  index=df[index_col])

    return y_split, label_enc


def get_weights_for_balanced_clf(y):
    """
    Gets sample weights for the WeightedRandomSampler() for the data loader to make balanced training datasets in each epoch.

    Let class_counts, shape (n_classes, ) be the vector of class counts. The sample weights for the ith observation is

    n_samples / class_counts[y[i]]

    Parameters
    ----------
    y: array-like, (n_samples, )
        The observed class indices.

    Output
    ------
    sample_weights: array-like, (n_samples, )
        The sample weights
    """

    y = pd.Series(y)
    class_counts = y.value_counts()  # class counts

    n_samples = len(y)

    sample_weights = n_samples / np.array([class_counts[cl_idx]
                                           for cl_idx in y])

    return sample_weights
