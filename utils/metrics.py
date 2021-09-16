import numpy as np
from typing import Dict, Tuple

from utils.inference_refactor import FeatureModel


def compute_log_posterior_predictive(test_observations: np.ndarray,
                                     inference_alg: FeatureModel,
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
    log_posterior_predictive_per_sample = np.zeros(num_samples)
    if inference_alg.model_str == 'linear_gaussian':
        for sample_idx in range(num_samples):
            sampled_posterior = inference_alg.sample_predictive_posterior()
            indicators_probs = sampled_posterior['indicators_probs']
            features = sampled_posterior['features']
            # if inference_alg_str == 'Doshi-Velez':
            #     indicators_probs = np.random.beta(
            #         a=variable_parameters['pi']['param_1'][:],
            #         b=variable_parameters['pi']['param_1'][:])
            #     A = np.stack([np.random.multivariate_normal(mean=variable_parameters['A']['mean'][k, :],
            #                                                 cov=variable_parameters['A']['cov'][k, :])
            #                   for k in range(len(indicators_probs))])
            # elif inference_alg_str == 'Widjaja':
            #     # TODO: investigate why some param_2 are negative and how to stop it.
            #     # Something in the original Widjaja code is screwing up.
            #     param_1 = variable_parameters['pi']['param_1']
            #     param_1[param_1 < 1e-10] = 1e-10
            #     param_2 = variable_parameters['pi']['param_2']
            #     param_2[param_2 < 1e-10] = 1e-10
            #
            #     indicators_probs = np.random.beta(a=param_1[-1, :], b=param_2[-1, :])
            #     A = np.stack([np.random.multivariate_normal(mean=variable_parameters['A']['mean'][-1, k, :],
            #                                                 cov=variable_parameters['A']['cov'][-1, k, :])
            #                   for k in range(len(indicators_probs))])
            # elif inference_alg_str == 'R-IBP':
            #     raise NotImplementedError
            # else:
            #     raise NotImplementedError

            # Treat each test observation as the "next" observation
            # shape: (num data, max num features)
            max_num_features = len(indicators_probs)
            Z = np.random.binomial(
                n=1,
                p=indicators_probs.reshape(1, -1),
                size=(num_obs, max_num_features))
            log_posterior_predictive_per_sample[sample_idx] = np.sum(
                np.square(test_observations - np.matmul(Z, features)))

        log_posterior_predictive_per_sample = -log_posterior_predictive_per_sample / (
                2.0 * inference_alg.model_params['gaussian_likelihood_cov_scaling'])
        log_posterior_predictive_per_sample -= obs_dim * np.log(
            2.0 * np.pi * inference_alg.model_params['gaussian_likelihood_cov_scaling']) / 2.

    else:
        raise NotImplementedError

    log_posterior_predictive_results = dict(
        mean=np.mean(log_posterior_predictive_per_sample),
        std=np.std(log_posterior_predictive_per_sample))

    return log_posterior_predictive_results
