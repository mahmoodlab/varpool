"""
Utilities for discrete time survival models.

Much of this code is adopted from https://github.com/mahmoodlab/Patch-GCN/blob/c6455a3a01c4ca20cde6ddb9a6f9cd807253a4f7/datasets/dataset_survival.py
"""
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_censored


def get_discrete_surv_bins(surv_df, n_bins, eps=1e-6,
                           time_col='survival_time', censor_col='censorship'):
    """
    Bins survival times into discrete time intervals and add corresponding bin labels to survival data frame.

    Parameters
    ----------
    surv_df: pd.DataFrame, (n_samples, n_features)
        The dataframe containing the surivial information.

    n_bins: int
        Number bins to bin time into.

    eps: float
        TODO: document.

    time_col: str
        Name of the time column.

    censor_col: str
        Name of the binary censorship column.

    Output
    ------
    time_bin_idx, bins

    time_bin_idx: array-like, (n_samples, )
        The index of the survival time bin label.

    bins: array-like, shape (n_bins + 1)
        The bin intervals cutoffs.
    """

    surv_df = surv_df.copy()

    # pull out all uncensored times
    censor_mask = surv_df[censor_col].astype(bool)
    uncensored_df = surv_df[~censor_mask]
    times_no_censor = uncensored_df[time_col]

    # TODO: document
    _, q_bins = pd.qcut(times_no_censor, q=n_bins, retbins=True, labels=False)
    q_bins[-1] = surv_df[time_col].max() + eps
    q_bins[0] = surv_df[time_col].min() - eps

    # y_discrete is the index label corresponding to the discrete time interval
    y_discr, bins = pd.cut(surv_df[time_col], bins=q_bins,
                           retbins=True, labels=False,
                           right=False, include_lowest=True)

    y_discr = y_discr.astype(int)

    return y_discr, bins

    # df.insert(loc=2, column='label', value=y_discr.values.astype(int))
    # surv_df['time_bin'] = y_discr

    # TODO: I don't think we actually need this -- can we get rid of it?
    # create label dictionary for bin X censorship class
    # bin_cen_label_dict = {}
    # key_count = 0
    # for bin_idx in range(len(q_bins)-1):
    #     for c in [0, 1]:
    #         bin_cen_label_dict.update({(bin_idx, c): key_count})
    #         key_count += 1
    # add index for bin X censorship class
    # for idx in surv_df.index:
    #     # add label for bins X censorship status
    #     bin_idx, c = surv_df.loc[idx, ['time_bin', censor_col]]
    #     surv_bin_cen_idx = bin_cen_label_dict[(bin_idx, int(c))]
    #     surv_df.loc[idx, 'time_bin_X_censor'] = surv_bin_cen_idx

    # format to ints
    # cols = ['time_bin', 'time_bin_X_censor']
    # surv_df[cols] = surv_df[cols].astype(int)
    # return surv_df, bins, bin_cen_label_dict


def dict_split_discr_surv_df(y_df, time_bin_col='time_bin',
                             time_col='survival_time',
                             censor_col='censorship',
                             index_col='sample_id',
                             split_col='split'):
    """
    Splits the discrete survival response data frame into train, validation, and test sets.

    Parameters
    ----------
    df: pd.DataFrame
        A data frame whose first four columns are  [sample_id, time_bin_col, censor_col, split_col]
        sample_id: is the identifier for each sample
        time_bin: is discrete time bin.
        censorship: is the censorship indicator
        split: should be one of ['train', 'val', 'test'] indicating which hold out set the sample is in

    time_bin_col: str
        Name of the time bin column.

    time_col: str
        Name of the survival time column.

    censor_col: str
        Name of the censorship column.

    index_col: str
        Name of the column containin the sample identifies; this will be used to index the pd.Series in y_split.

    spilt_col: str
        Name of the column containing the train/test/val splits.

    Output
    ------
    y_split

    y_split: dict of pd.DataFrame
        The keys include ['train', 'val', 'test']. The data frames are indexed by the index_col and contain columns ['time_bin', 'censorship', 'survival_time'].
    """

    # pull out the important columns and copy.
    cols = [index_col, time_bin_col, censor_col, time_col, split_col]
    y_df = y_df[cols].copy()

    # tranform class names to indices

    # split into train/val/test sets
    y_split = {}
    for kind, df in y_df.groupby(split_col):
        assert kind in ['train', 'val', 'test']

        ##########
        # format #
        ##########

        # subset and set index
        df = df[[index_col, time_bin_col, censor_col, time_col]].\
            set_index(index_col)

        # format numbers
        ensure_int_cols = [time_bin_col, censor_col]
        df[ensure_int_cols] = df[ensure_int_cols].astype(int)

        # standardize names
        df = df.rename(columns={time_bin_col: 'time_bin',
                                censor_col: 'censorship',
                                time_col: 'survival_time'}
                       )

        y_split[kind] = df

    return y_split


def get_weights_for_balanced_binXcensor(surv_df,
                                        time_bin_col='time_bin',
                                        censor_col='censorship'):
    """
    Gets sample weights for the WeightedRandomSampler() for the data loader used by a discrete survival task.

    - let tb_X_c, shape (n_samples, ) be the labels for the time bin X censorship status classes
    - let tb_X_c_counts, shape (n_classes_tb_X_c, ) be the counts of number of sampels in each of these classes

    the the sample weight for sample i is given by

    n_samples / tb_X_c_counts[tb_X_c[i]]

    Parameters
    ----------
    surv_df: pd.DataFrame, (n_samples, 2)
        The survival response data frame containing the time bin columns and censorship status.

    time_bin_col: str
        Name of the time bin column.

    censor_col: str
        Name of the censorship column.

    Output
    ------
    sample_weights: array-like, (n_samples, )
        The sample weights
    """
    # This is implementing https://github.com/mahmoodlab/Patch-GCN/blob/c6455a3a01c4ca20cde6ddb9a6f9cd807253a4f7/utils/utils.py#L184

    # create time_bin X censorship status labels
    tb_X_c = surv_df[time_bin_col].astype(str)\
        + '_X_' \
        + surv_df[censor_col].astype(str)

    tb_X_c_counts = tb_X_c.value_counts()  # class counts

    # class counts for each sample
    sample_tbXc_couts = np.array([tb_X_c_counts[label]
                                 for label in tb_X_c])

    n_samples = len(tb_X_c)

    sample_weights = n_samples / sample_tbXc_couts

    return sample_weights


def get_perm_c_index_quantile(event, time, n_perm=1000, q=0.95):
    """
    Gets the qth quantile from the permutation distribution of the c-index.

    Parameters
    ----------
    event : array-like, shape = (n_samples,)
        Boolean array denotes whether an event occurred.

    time : array-like, shape = (n_samples,)
        Array containing the time of an event or time of censoring.

    n_perm: int
        Number of permutation samples to draw.

    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.

    Output
    ------
    quantiles: float
        The qth quantile of the permutation distribution.
    """

    perm_samples = []
    random_estimate = np.arange(len(time))
    for _ in range(n_perm):

        # randomly permuted estimate!
        random_estimate = np.random.permutation(random_estimate)

        ci_perm = concordance_index_censored(event_indicator=event,
                                             event_time=time,
                                             estimate=random_estimate)[0]

        perm_samples.append(ci_perm)

    return np.quantile(a=perm_samples, q=q)
