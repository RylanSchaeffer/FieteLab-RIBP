import numpy as np
from typing import Dict, Tuple

import scipy.stats

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

    sampled_param_posterior = inference_alg.sample_variables_for_predictive_posterior(
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

    sampled_variables_posterior = inference_alg.sample_variables_for_predictive_posterior(
        num_samples=num_samples)

    # shape: (max num features,)
    indicators_probs = sampled_variables_posterior['indicators_probs']
    assert len(indicators_probs.shape) == 1
    max_num_features = indicators_probs.shape[0]

    # Treat each test observation as the "next" observation
    # shape (num samples, num obs, max num features)
    test_Z = np.random.binomial(
        n=1,
        p=indicators_probs.reshape(1, max_num_features),
        size=(num_samples, num_test_obs, max_num_features))

    sigma_x_sqrd = np.square(inference_alg.gen_model_params['likelihood_params']['sigma_x'])
    sigma_A_sqrd = inference_alg.gen_model_params['feature_prior_params']['feature_prior_cov_scaling']

    # If the inference algorithm infers features, use those
    if 'features' in sampled_variables_posterior:

        # shape: (num samples, max num features, obs dim)
        features = sampled_variables_posterior['features']

        # shape = (num samples, num obs, obs dim)
        pred_means = np.einsum(
            'sok,skd->sod',  # s=samples, o=observations, k=features, d=observations dimension
            test_Z,  # shape: samples, observations, num features
            features,  # shape: samples, num features, obs dim
        )
        # shape (num samples,)
        log_posterior_predictive_per_sample = np.sum(
            np.sum(np.square(test_observations.reshape(1, num_test_obs, obs_dim) - pred_means),
                   axis=1),
            axis=1)

        log_posterior_predictive_per_sample = -log_posterior_predictive_per_sample / (
                2.0 * sigma_x_sqrd)
        log_posterior_predictive_per_sample -= obs_dim * np.log(
            2.0 * np.pi * sigma_x_sqrd) / 2.

    # If the inference algorithm doesn't infer features, use
    # calculated posterior predictive algorithm
    else:

        # shape: (num train observations, max num features)
        train_Z = sampled_variables_posterior['indicators_probs_train']
        # shape: (num samples, num train obs, max num features)
        train_Z = np.repeat(train_Z[np.newaxis], axis=0, repeats=num_samples)

        # shape: (num train observations, obs dim)
        observations_train = sampled_variables_posterior['observations_train']
        num_train_obs = observations_train.shape[0]

        log_posterior_predictive_per_sample = np.zeros(shape=(num_samples,))

        log_prior_prob = sampled_variables_posterior['log_prior_prob']
        log_posterior_predictive_per_sample += log_prior_prob

        # shape: num train + num test, obs dim
        Otilde = np.concatenate([test_observations, train_observations])
        # shape (num samples, num train + num test, max num features)
        Ztilde = np.concatenate([test_Z, train_Z], axis=1)

        V = sigma_x_sqrd * np.eye(obs_dim)
        # Vinv = np.linalg.inv(V)
        for sample_idx in range(num_samples):

            sample_Ztilde = Ztilde[sample_idx]
            # (num train + num test, num train + num test)
            Uinv = np.eye(num_train_obs + num_test_obs) - np.einsum(
                'ab,bc,cd',
                sample_Ztilde,
                np.linalg.inv(sample_Ztilde.T @ sample_Ztilde + sigma_x_sqrd * np.eye(max_num_features) / sigma_A_sqrd),
                sample_Ztilde.T)
            U = np.linalg.inv(Uinv)
            U_testtest = U[:num_test_obs, :num_test_obs]  # shape(num test, num test)
            U_testtrain = U[:num_test_obs, num_test_obs:]  # shape(num test, num train)
            U_traintest = U[num_test_obs:, :num_test_obs]  # shape(num train, num test)
            U_traintrain = U[num_test_obs:, num_test_obs:]  # shape(num train, num train)

            Ztilde_test = Ztilde[sample_idx, :num_test_obs]
            Ztilde_train = train_Z[sample_idx, num_test_obs:]
            Otilde_test = Otilde[:num_test_obs]
            Otilde_train = Otilde[num_test_obs:]

            # shapes of each term:
            #   (obs dim * num test data, obs dim * num train data)
            #   * (obs dim * num train data, obs dim * num train data)
            #   * (obs dim * num train data)
            # Total shape: (num test * obs dim)

            inv_kron_V_Utraintrain = np.linalg.inv(np.kron(V, U_traintrain))

            mean = np.kron(V, U_testtrain) \
                   @ inv_kron_V_Utraintrain \
                   @ Otilde_train.flatten('F')

            cov = np.kron(V, U_testtest)\
                  - np.kron(V, U_testtrain) @ inv_kron_V_Utraintrain @ np.kron(V, U_traintest)

            sample_log_posterior_predictive = scipy.stats.multivariate_normal.logpdf(
                x=Otilde_test.reshape(-1),
                mean=mean,
                cov=cov,
            )

            log_posterior_predictive_per_sample[sample_idx] = sample_log_posterior_predictive

    log_posterior_predictive_results = dict(
        mean=np.mean(log_posterior_predictive_per_sample),
        std=np.std(log_posterior_predictive_per_sample))

    return log_posterior_predictive_results


def compute_reconstruction_error_linear_gaussian(observations: np.ndarray,
                                                 dish_eating_posteriors: np.ndarray,
                                                 features_after_last_obs: np.ndarray,
                                                 ) -> float:
    """
    Compute the reconstruction error: ||X - ZA||_2^2.
    """

    if features_after_last_obs is not None:
        diff = observations - dish_eating_posteriors @ features_after_last_obs
        error = np.square(np.linalg.norm(diff))
    else:
        error = np.nan
    return error

