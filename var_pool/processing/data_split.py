from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import numpy as np

# TODO: test code to add
# n_samples = 99
# train_idxs, val_idxs, test_idxs = train_test_val_split(n_samples)
# assert len(train_idxs) + len(val_idxs) + len(test_idxs) == n_samples
def train_test_val_split(n_samples,
                         train_size=0.8, val_size=0.1, test_size=0.1,
                         shuffle=True, random_state=None, stratify=None):
    """
    Creates the indices for a train/validation/test set.
    
    Parameters
    ----------
    samples: int, array-like
        Either the total number of samples or an array containing the sample identifiers.
        
    train_size, val_size, test_size: float
        The train/validation/test proportions. Need to add to 1.
        
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
        
    Output
    ------
    train_idxs, val_idxs, test_idxs
    """

    # ssert all([train_size > 0, val_size > 0, test_size > 0])
    # assert np.allclose(train_size + val_size + test_size, 1)
    rng = check_random_state(random_state)

    idxs = np.arange(int(n_samples))
    # # user input list of samples
    # if not isinstance(samples, Number):
    #     samples = np.array(samples).reshape(-1)
    #     idxs = np.arange(len(samples))
    # else:
    #     idxs = np.arange(int(samples))

    if test_size == 0:
        train_idxs, val_idxs = train_test_split(idxs,
                                                train_size=train_size,
                                                test_size=val_size,
                                                random_state=rng,
                                                shuffle=shuffle,
                                                stratify=stratify)

        test_idxs = []

        return train_idxs, val_idxs, test_idxs

    # split test set off
    tr_val_idxs, test_idxs = train_test_split(idxs,
                                              train_size=train_size + val_size,
                                              test_size=test_size,
                                              random_state=rng,
                                              shuffle=shuffle,
                                              stratify=stratify)
    
    if stratify is not None:
        stratify = stratify[tr_val_idxs]

    # calculate number of train samples
    n_train = len(tr_val_idxs) * (train_size / (train_size + val_size))
    n_train = int(n_train)

    # split train and validation set
    train_idxs, val_idxs = train_test_split(tr_val_idxs,
                                            train_size=n_train,
                                            random_state=rng, shuffle=shuffle,
                                            stratify=stratify)

    return train_idxs, val_idxs, test_idxs
