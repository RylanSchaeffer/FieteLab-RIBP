import numpy as np


def assert_no_nan_no_inf(x):
    assert np.all(~np.isnan(x))
    assert np.all(~np.isinf(x))


def logits_to_probs(logits):
    probs = 1. / (1. + np.exp(-logits))
    return probs


def probs_to_logits(probs):
    logits = - np.log(1. / probs - 1.)
    return logits
