import numpy as np
from typing import Dict, Tuple


def predictive_log_likelihood(test_observations: np.ndarray,
                              inference_alg_str: str,
                              likelihood_model: str,
                              variable_parameters: Dict[str, dict],
                              model_parameters: Dict[str, np.ndarray],
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
    log_likelihoods_per_sample = np.zeros(num_samples)
    if likelihood_model == 'linear_gaussian':
        for sample_idx in range(num_samples):
            if inference_alg_str == 'Doshi-Velez':
                sticks = np.random.beta(a=variable_parameters['v']['param_1'][:],
                                        b=variable_parameters['v']['param_1'][:])
                indicators_probs = np.cumprod(sticks)
                A = np.stack([np.random.multivariate_normal(mean=variable_parameters['A']['mean'][k, :],
                                                            cov=variable_parameters['A']['cov'][k, :])
                              for k in range(max_num_features)])

            elif inference_alg_str == 'HMC-Gibbs':
                random_mcmc_sample_idx = np.random.choice(
                    np.arange(variable_parameters['v']['value'].shape[0]))
                indicators_probs = variable_parameters['v']['value'][random_mcmc_sample_idx, :]
                A = variable_parameters['A']['mean'][random_mcmc_sample_idx, :]

            elif inference_alg_str == 'Widjaja':
                # TODO: investigate why some param_2 are negative and how to stop it.
                # Something in the original Widjaja code is screwing up.
                param_1 = variable_parameters['v']['param_1']
                param_1[param_1 < 1e-10] = 1e-10
                param_2 = variable_parameters['v']['param_2']
                param_2[param_2 < 1e-10] = 1e-10

                sticks = np.random.beta(a=param_1[-1, :], b=param_2[-1, :])
                indicators_probs = np.cumprod(sticks)
                A = np.stack([np.random.multivariate_normal(mean=variable_parameters['A']['mean'][-1, k, :],
                                                            cov=variable_parameters['A']['cov'][-1, k, :])
                              for k in range(len(indicators_probs))])
            elif inference_alg_str == 'R-IBP':
                raise NotImplementedError
            else:
                raise NotImplementedError

            # Treat each test observation as the "next" observation
            # shape: (num data, max num features)
            max_num_features = len(indicators_probs)
            Z = np.random.binomial(
                n=1,
                p=indicators_probs.reshape(1, -1),
                size=(num_obs, max_num_features))

            log_likelihoods_per_sample[sample_idx] = np.sum(np.square(test_observations - np.matmul(Z, A)))

        log_likelihoods_per_sample = -log_likelihoods_per_sample / (
                    2.0 * model_parameters['gaussian_likelihood_cov_scaling'])
        log_likelihoods_per_sample -= obs_dim * np.log(
            2.0 * np.pi * model_parameters['gaussian_likelihood_cov_scaling']) / 2.

    else:
        raise NotImplementedError

    return np.mean(log_likelihoods_per_sample), np.std(log_likelihoods_per_sample)
