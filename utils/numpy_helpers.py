import numpy as np


def assert_no_nan_no_inf(x):
    assert np.all(~np.isnan(x))
    assert np.all(~np.isinf(x))


def compute_largest_dish_idx(observations: np.ndarray,
                             cutoff: float = 1e-4):
    """
    Compute the highest column index such that the column has at least one
    value >= cutoff.

    Array assumed to have shape (num obs, max num features)
    """

    observations_copy = np.copy(observations)
    observations_copy = observations_copy.astype(np.float)
    column_sums = np.sum(observations >= cutoff, axis=0)
    columns_has_value_gt_cutoff = column_sums > 0
    false_columns = np.argwhere(columns_has_value_gt_cutoff == False)
    if len(false_columns) == 0:
        return observations_copy.shape[1]
    else:
        return false_columns[0, 0]


def logits_to_probs(logits):
    probs = 1. / (1. + np.exp(-logits))
    return probs


def probs_to_logits(probs):
    logits = - np.log(1. / probs - 1.)
    return logits
