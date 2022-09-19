from torch.utils.data import Sampler
from itertools import combinations
import numpy as np

# TODO: add shuffle
class ComparablePairSampler(Sampler):
    """
    Iterates over comparable pairs.

    Parameters
    ----------
    times: array-like, (n_samples, )
        The survival times.

    censor: array-like, (n_samples, )
        The censor indicators.

    """
    def __init__(self, times, censor):
        events = ~censor.astype(bool)
        self.pairs = get_comparable_pairs(times=times, events=events)

    def __iter__(self):
        return iter(self.pairs)

    def __len__(self):
        return len(self.pairs)


def get_comparable_pairs(times, events):
    """
    Gets the comparable pairs.

    Parmeters
    ---------
    times: array-like, (n_samples, )
        The survival times.

    events: array-like, (n_samples, )
        The event indicators.

    Output
    ------
    pairs: list, (n_comparable, 2)
        The indices of the comparable pairs where the first entry is the more risky sample (smaller survival).
    """
    times = np.array(times).reshape(-1)
    events = np.array(events).reshape(-1)

    events = events.astype(bool)
    n_samples = times.shape[0]

    pairs = []
    for (idx_a, idx_b) in combinations(range(n_samples), 2):
        time_a, event_a = times[idx_a], events[idx_a]
        time_b, event_b = times[idx_b], events[idx_b]

        if time_a < time_b and event_a:
            # a and b are comparable, a is more risky
            pairs.append([idx_a, idx_b])

        elif time_b < time_a and event_b:
            # a and b are comparable, b is more risky
            pairs.append([idx_b, idx_a])

    return pairs
