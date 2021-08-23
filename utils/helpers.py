import numpy as np
import torch


def assert_numpy_no_nan_no_inf(x):
    assert np.all(~np.isnan(x))
    assert np.all(~np.isinf(x))


def assert_torch_no_nan_no_inf(x):
    assert torch.all(~torch.isnan(x))
    assert torch.all(~torch.isinf(x))


def numpy_logits_to_probs(logits):
    probs = 1. / (1. + np.exp(-logits))
    return probs


def numpy_probs_to_logits(probs):
    logits = - np.log(1. / probs - 1.)
    return logits


def torch_expected_log_bernoulli_prob(p_prob: torch.Tensor,
                                      q_prob: torch.Tensor) -> torch.Tensor:
    """
    Compute E_{q(x)}[log p(x)] where p(x) and q(x)

    All inputs are expected to have a batch dimension, then appropriate shape dimensions
    i.e. if p_probs is (batch, 3), then q_probs should be (batch, 3).
    """

    one_minus_p_probs = 1. - p_prob
    term_one = torch.sum(torch.multiply(q_prob,
                                        torch.log(torch.divide(p_prob,
                                                               one_minus_p_probs))))
    term_two = torch.sum(torch.log(one_minus_p_probs))
    total = term_one + term_two
    return total


def torch_expected_log_gaussian_prob(p_mean: torch.Tensor,
                                     p_cov: torch.Tensor,
                                     q_mean: torch.Tensor,
                                     q_cov: torch.Tensor) -> torch.Tensor:
    """
    Compute E_{q(x)}[log p(x)] where p(x) and q(x) are both Gaussian.

    All inputs are expected to have a batch dimension, then appropriate shape dimensions
    i.e. if p_mean is (batch, 3), then q_mean should be (batch, 3) and p_cov and q_cov
    should both be (batch, 3, 3).
    """

    # gaussian_dim = p_mean.shape[-1]
    batch_dim = p_mean.shape[0]

    # sum_k -0.5 * log (2 pi |p_cov|)
    term_one = -0.5 * torch.sum(2 * np.pi * torch.linalg.det(p_cov))
    p_precision = torch.linalg.inv(p_cov)  # recall, precision = cov^{-1}

    # sum_k -0.5 Tr[p_precision (q_cov + q_mean q_mean^T)]
    term_two = -0.5 * torch.sum(
        torch.einsum(
            'bii->b',
            torch.einsum(
                'bij, bjk->bik',
                p_precision,
                q_cov + torch.einsum('bi, bj->bij',
                                     q_mean,
                                     q_mean))))
    term_two_check = -0.5 * torch.sum(torch.stack(
        [torch.trace(torch.matmul(p_precision[k],
                                  torch.add(q_cov[k],
                                            torch.outer(q_mean[k], q_mean[k].T))))
         for k in range(batch_dim)]))
    assert torch.isclose(term_two, term_two_check)

    # sum_k -0.5 * -2 * mean_p Precision_p mean_q
    term_three = -0.5 * -2. * torch.einsum(
        'bi,bij,bj',
        p_mean,
        p_precision,
        q_mean,
    )
    term_three_check = torch.sum(torch.stack(
        [-.5 * -2. * torch.inner(p_mean[k],
                                 torch.matmul(p_precision[k],
                                              q_mean[k]))
         for k in range(batch_dim)]))
    assert torch.isclose(term_three, term_three_check)

    # sum_k -0.5 * mean_p Precision_p mean_p
    term_four = -0.5 * -2. * torch.einsum(
        'bi,bij,bj',
        p_mean,
        p_precision,
        p_mean,
    )
    term_four_check = torch.sum(torch.stack(
        [-.5 * -2. * torch.inner(p_mean[k],
                                 torch.matmul(p_precision[k],
                                              p_mean[k]))
         for k in range(batch_dim)]))
    assert torch.isclose(term_four, term_four_check)
    total = term_one + term_two + term_three + term_four
    return total


def torch_expected_log_likelihood(observation: torch.Tensor,
                                  ) -> torch.Tensor:
    raise NotImplementedError


def torch_entropy_bernoulli(probs: torch.Tensor) -> torch.Tensor:
    one_minus_probs = 1. - probs
    log_odds = torch.log(torch.divide(probs, one_minus_probs))
    entropy = -1. * torch.sum(torch.add(torch.multiply(probs,
                                                     log_odds),
                                      torch.log(one_minus_probs)))
    assert entropy >= 0.
    return entropy


def torch_entropy_gaussian(mean: torch.Tensor,
                           cov: torch.Tensor) -> torch.Tensor:
    dim = mean.shape[-1]

    entropy = 0.5 * torch.sum(
        dim + dim * torch.log(torch.full(fill_value=2., size=()) * np.pi)
        + torch.log(torch.linalg.det(cov)))
    assert entropy >= 0.
    return entropy


def torch_logits_to_probs(logits):
    probs = 1. / (1. + torch.exp(-logits))
    return probs


def torch_probs_to_logits(probs):
    logits = - torch.log(1. / probs - 1.)
    return logits

