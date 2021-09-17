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


def convert_half_cov_to_cov(half_cov: np.ndarray) -> np.ndarray:
    """
    Converts half-covariance M into covariance M^T M in a batched manner.
    Convert batch half-covariance to covariance.

    Assumes M is [batch size, D, D] tensor. If no batch dimension exists,
    one will be added and then subsequently removed.
    """

    # if no batch dimension is given, add one
    has_batch_dim = len(half_cov.shape) == 3
    if not has_batch_dim:
        half_cov = np.expand_dims(half_cov, dim=0)

    cov = np.einsum(
        'abc, abd->acd',
        half_cov,
        half_cov)
    # batch_size = half_cov.shape[0]
    # cov_test = torch.stack([torch.matmul(half_cov[k].T, half_cov[k])
    #                         for k in range(batch_size)])
    # assert torch.allclose(cov, cov_test)

    # if no batch dimension was originally given, remove
    if not has_batch_dim:
        cov = np.squeeze(cov, dim=0)

    assert_no_nan_no_inf(cov)
    return cov


def logits_to_probs(logits):
    probs = 1. / (1. + np.exp(-logits))
    return probs


def probs_to_logits(probs):
    logits = - np.log(1. / probs - 1.)
    return logits
