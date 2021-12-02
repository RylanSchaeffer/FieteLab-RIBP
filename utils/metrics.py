import numpy as np
from typing import Dict, Tuple

from utils.prob_models.linear_gaussian import LinearGaussianModel


def compute_log_posterior_predictive(train_observations: np.ndarray,
                                     test_observations: np.ndarray,
                                     inference_alg,
                                     num_samples: int = 100,
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    """

    if inference_alg.model_str == 'linear_gaussian':
        log_posterior_predictive_results = compute_log_posterior_predictive_linear_gaussian(
            train_observations=train_observations,
            test_observations=test_observations,
            inference_alg=inference_alg,
            num_samples=num_samples)
    elif inference_alg.model_str == 'factor_analysis':
        log_posterior_predictive_results = compute_log_posterior_predictive_factor_analysis(
            train_observations=train_observations,
            test_observations=test_observations,
            inference_alg=inference_alg,
            num_samples=num_samples)
    elif inference_alg.model_str == 'nonnegative_matrix_factorization':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return log_posterior_predictive_results


def compute_log_posterior_predictive_factor_analysis(train_observations: np.ndarray,
                                                     test_observations: np.ndarray,
                                                     inference_alg: LinearGaussianModel,
                                                     num_samples: int = 100,
                                                     ) -> Tuple[np.ndarray, np.ndarray]:
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

    # shape: (num samples, max num features)
    scales = sampled_param_posterior['scales']

    # shape: (num samples, max num features, obs dim)
    features = sampled_param_posterior['features']

    # Treat each test observation as the "next" observation
    max_num_features = indicators_probs.shape[1]
    # shape: (num data, num obs, max num features)
    Z = np.random.binomial(
        n=1,
        p=indicators_probs.reshape(num_samples, 1, max_num_features),
        size=(num_samples, num_obs, max_num_features))

    # shape: (num data, num obs, max num features)
    scales_repeated = np.repeat(
        scales.reshape(num_samples, 1, max_num_features),
        repeats=num_obs,
        axis=1)
    # shape = (num samples, num obs, obs dim)
    pred_means = np.einsum(
        'sok,skd->sod',  # s=samples, o=observations, k=features, d=observations dimension
        np.multiply(Z, scales_repeated),
        features)
    # shape (num samples,)
    log_posterior_predictive_per_sample = np.sum(
        np.sum(np.square(test_observations.reshape(1, num_obs, obs_dim) - pred_means),
               axis=1),
        axis=1)

    log_posterior_predictive_per_sample = -log_posterior_predictive_per_sample / (
            2.0 * inference_alg.gen_model_params['likelihood_params']['sigma_x'] ** 2)
    log_posterior_predictive_per_sample -= obs_dim * np.log(
        2.0 * np.pi * inference_alg.gen_model_params['likelihood_params']['sigma_x'] ** 2) / 2.

    log_posterior_predictive_results = dict(
        mean=np.mean(log_posterior_predictive_per_sample),
        std=np.std(log_posterior_predictive_per_sample))

    return log_posterior_predictive_results


def compute_log_posterior_predictive_linear_gaussian(train_observations: np.ndarray,
                                                     test_observations: np.ndarray,
                                                     inference_alg: LinearGaussianModel,
                                                     num_samples: int = 100,
                                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the predictive log likelihood of new data using a Monte Carlo estimate:

    The predictive likelihood is defined as:
        p(X_{test} | X_{train})
            = \int p(X_{test} | Z_{test}, A) p(Z_{test}, A | X_{train})
            \approx \sum_{Z, A \sim p(Z_{test}, A | X_{train})} p(X_{test} | Z_{test}, A)

    Indicator probs should be calculated using the test observations.
    """

    num_train_obs, _ = train_observations.shape
    num_test_obs, obs_dim = test_observations.shape

    sampled_param_posterior = inference_alg.sample_params_for_predictive_posterior(
        num_samples=num_samples)
    # shape: (num samples, max num features)
    indicators_probs = sampled_param_posterior['indicators_probs']
    max_num_features = indicators_probs.shape[1]

    # If the inference algorithm infers features, use those
    if 'features' in sampled_param_posterior:

        # shape: (num samples, max num features, obs dim)
        features = sampled_param_posterior['features']

        # Treat each test observation as the "next" observation
        # shape: (num data, max num features)
        Z = np.random.binomial(
            n=1,
            p=indicators_probs.reshape(num_samples, 1, max_num_features),
            size=(num_samples, num_test_obs, max_num_features))  # shape (num samples, num obs, max num features)
        # shape = (num samples, num obs, obs dim)
        pred_means = np.einsum(
            'sok,skd->sod',  # s=samples, o=observations, k=features, d=observations dimension
            Z,
            features)
        # shape (num samples,)
        log_posterior_predictive_per_sample = np.sum(
            np.sum(np.square(test_observations.reshape(1, num_test_obs, obs_dim) - pred_means),
                   axis=1),
            axis=1)

        log_posterior_predictive_per_sample = -log_posterior_predictive_per_sample / (
                2.0 * (inference_alg.gen_model_params['likelihood_params']['sigma_x'] ** 2))
        log_posterior_predictive_per_sample -= obs_dim * np.log(
            2.0 * np.pi * (inference_alg.gen_model_params['likelihood_params']['sigma_x'] ** 2)) / 2.

        log_posterior_predictive_results = dict(
            mean=np.mean(log_posterior_predictive_per_sample),
            std=np.std(log_posterior_predictive_per_sample))

    # If the inference algorithm doesn't infer features, use
    # Griffiths & Ghahramani 2011
    else:
        raise NotImplementedError

    return log_posterior_predictive_results


def compute_reconstruction_error_linear_gaussian(observations: np.ndarray,
                                                 dish_eating_posteriors: np.ndarray,
                                                 features_after_last_obs: np.ndarray,
                                                 ) -> float:
    """
    Compute the reconstruction error: ||X - ZA||_2^2.
    """

    diff = observations - dish_eating_posteriors @ features_after_last_obs
    error = np.square(np.linalg.norm(diff))
    return error

