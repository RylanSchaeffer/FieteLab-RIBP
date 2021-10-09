import numpy as np
from typing import Dict, Tuple

from utils.inference import LinearGaussianModel


def compute_log_posterior_predictive(test_observations: np.ndarray,
                                     inference_alg,
                                     num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    """

    if inference_alg.model_str == 'linear_gaussian':
        log_posterior_predictive_results = compute_log_posterior_predictive_linear_gaussian(
            test_observations=test_observations,
            inference_alg=inference_alg,
            num_samples=num_samples)
    elif inference_alg.model_str == 'factor_analysis':
        raise NotImplementedError
    elif inference_alg.model_str == 'nonnegative_matrix_factorization':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return log_posterior_predictive_results


def compute_log_posterior_predictive_factor_analysis(test_observations: np.ndarray,
                                                     inference_alg: LinearGaussianModel,
                                                     num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the predictive log likelihood of new data using a Monte Carlo estimate:

    The predictive likelihood is defined as:
        p(X_{test} | X_{train})
            = \int p(X_{test} | Z_{test}, A) p(Z_{test}, A | X_{train})
            \approx \sum_{Z, A \sim p(Z_{test}, A | X_{train})} p(X_{test} | Z_{test}, A)

    Indicator probs should be calculated using the test observations.
    """

    num_obs, obs_dim = test_observations.shape

    sampled_param_posterior = inference_alg.sample_params_for_predictive_posterior(
        num_samples=num_samples)
    # shape: (num samples, max num features)
    indicators_probs = sampled_param_posterior['indicators_probs']
    # shape: (num samples, max num features, obs dim)
    features = sampled_param_posterior['features']

    # Treat each test observation as the "next" observation
    # shape: (num data, max num features)
    max_num_features = indicators_probs.shape[1]
    Z = np.random.binomial(
        n=1,
        p=indicators_probs.reshape(num_samples, 1, max_num_features),
        size=(num_samples, num_obs, max_num_features))  # shape (num samples, num obs, max num features)
    # shape = (num samples, num obs, obs dim)
    pred_means = np.einsum(
        'sok,skd->sod',  # s=samples, o=observations, k=features, d=observations dimension
        Z,
        features)
    # shape (num samples,)
    log_posterior_predictive_per_sample = np.sum(
        np.sum(np.square(test_observations.reshape(1, num_obs, obs_dim) - pred_means),
               axis=1),
        axis=1)

    log_posterior_predictive_per_sample = -log_posterior_predictive_per_sample / (
            2.0 * inference_alg.gen_model_params['gaussian_likelihood_cov_scaling'])
    log_posterior_predictive_per_sample -= obs_dim * np.log(
        2.0 * np.pi * inference_alg.gen_model_params['gaussian_likelihood_cov_scaling']) / 2.

    log_posterior_predictive_results = dict(
        mean=np.mean(log_posterior_predictive_per_sample),
        std=np.std(log_posterior_predictive_per_sample))

    return log_posterior_predictive_results


def compute_log_posterior_predictive_linear_gaussian(test_observations: np.ndarray,
                                                     inference_alg: LinearGaussianModel,
                                                     num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the predictive log likelihood of new data using a Monte Carlo estimate:

    The predictive likelihood is defined as:
        p(X_{test} | X_{train})
            = \int p(X_{test} | Z_{test}, A) p(Z_{test}, A | X_{train})
            \approx \sum_{Z, A \sim p(Z_{test}, A | X_{train})} p(X_{test} | Z_{test}, A)

    Indicator probs should be calculated using the test observations.
    """

    num_obs, obs_dim = test_observations.shape

    sampled_param_posterior = inference_alg.sample_params_for_predictive_posterior(
        num_samples=num_samples)
    # shape: (num samples, max num features)
    indicators_probs = sampled_param_posterior['indicators_probs']
    # shape: (num samples, max num features, obs dim)
    features = sampled_param_posterior['features']

    # Treat each test observation as the "next" observation
    # shape: (num data, max num features)
    max_num_features = indicators_probs.shape[1]
    Z = np.random.binomial(
        n=1,
        p=indicators_probs.reshape(num_samples, 1, max_num_features),
        size=(num_samples, num_obs, max_num_features))  # shape (num samples, num obs, max num features)
    # shape = (num samples, num obs, obs dim)
    pred_means = np.einsum(
        'sok,skd->sod',  # s=samples, o=observations, k=features, d=observations dimension
        Z,
        features)
    # shape (num samples,)
    log_posterior_predictive_per_sample = np.sum(
        np.sum(np.square(test_observations.reshape(1, num_obs, obs_dim) - pred_means),
               axis=1),
        axis=1)

    log_posterior_predictive_per_sample = -log_posterior_predictive_per_sample / (
            2.0 * (inference_alg.gen_model_params['likelihood_params']['sigma_x'] ** 2))
    log_posterior_predictive_per_sample -= obs_dim * np.log(
        2.0 * np.pi * (inference_alg.gen_model_params['likelihood_params']['sigma_x'] ** 2)) / 2.

    log_posterior_predictive_results = dict(
        mean=np.mean(log_posterior_predictive_per_sample),
        std=np.std(log_posterior_predictive_per_sample))

    return log_posterior_predictive_results
