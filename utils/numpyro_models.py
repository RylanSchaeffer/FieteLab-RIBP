import jax.numpy as jnp
import numpyro
from typing import Dict


def create_factor_analysis_model(model_params: Dict[str, float],
                                 max_num_features: int,
                                 num_obs: int,
                                 obs_dim: int):
    def factor_analysis_model(obs):
        raise NotImplementedError

    return factor_analysis_model


def create_linear_gaussian_model(model_params: Dict[str, float],
                                 max_num_features: int,
                                 num_obs: int,
                                 obs_dim: int):

    # TODO: figure out why this prevents HMC from pickling
    def linear_gaussian_model(obs):
        with numpyro.plate('stick_plate', max_num_features):
            # Based on Ghahramani 2007 Bayesian Nonparametric Latent Feature
            # Models. They sample the number of new dishes as:
            #       pi_k \sim Beta(alpha * beta / K, beta)
            sticks = numpyro.sample(
                'sticks',
                numpyro.distributions.Beta(
                    model_params['IBP']['alpha'] * model_params['IBP']['beta'] / max_num_features,
                    model_params['IBP']['beta']))

        # For each feature, sample its value
        with numpyro.plate('features_plate', max_num_features):
            features = numpyro.sample(
                'A',
                numpyro.distributions.MultivariateNormal(
                    loc=jnp.zeros(obs_dim),
                    covariance_matrix=model_params['feature_prior_params'][
                                          'gaussian_mean_prior_cov_scaling'] * jnp.eye(obs_dim)))

        with numpyro.plate('data', num_obs):
            # For some reason, this broadcasting is easier with numpyro. Don't fight it.
            # shape (max num features, num obs)
            indicators_transposed = numpyro.sample(
                'Z',
                numpyro.distributions.Bernoulli(probs=sticks.reshape(-1, 1)))
            numpyro.sample(
                'obs',
                numpyro.distributions.MultivariateNormal(
                    loc=jnp.matmul(indicators_transposed.T, features),
                    covariance_matrix=model_params['likelihood_params']['sigma_x'] * jnp.eye(obs_dim)),
                obs=obs)

    return linear_gaussian_model


def create_nonnegative_matrix_factorization_model(model_params: Dict[str, float],
                                                  max_num_features: int,
                                                  num_obs: int,
                                                  obs_dim: int):
    def nonnegative_matrix_factorization_model(obs):
        raise NotImplementedError

    return nonnegative_matrix_factorization_model
