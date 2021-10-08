import abc
import cvxpy as cp
import jax
import jax.random
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import os
import scipy.linalg
from scipy.stats import poisson
from timeit import default_timer as timer
import torch
import tracemalloc
from typing import Dict, Tuple, Union

import utils.numpy_helpers
import utils.numpyro_models
import utils.torch_helpers
import utils.inference_widjaja

# torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_default_tensor_type('torch.FloatTensor')

inference_algs = [
    'HMC-Gibbs',
    'Doshi-Velez',
    'R-IBP',
    'Widjaja',
]


class LinearGaussianModel(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.model_str = None
        self.model_params = None
        self.plot_dir = None
        self.fit_results = None

    @abc.abstractmethod
    def fit(self,
            observations: np.ndarray):
        pass

    @abc.abstractmethod
    def sample_params_for_predictive_posterior(self,
                                               num_samples: int):
        pass

    @abc.abstractmethod
    def features_after_last_obs(self) -> np.ndarray:
        """
        Returns array of shape (num features, feature dimension)
        """
        pass

    @abc.abstractmethod
    def features_by_obs(self) -> np.ndarray:
        """
        Returns array of shape (num obs, num features, feature dimension)
        :return:
        """
        pass


class DoshiVelezLinearGaussian(LinearGaussianModel):
    """
    Implementation of Doshi-Velez 2009 Variational Inference for the IBP.
    """

    def __init__(self,
                 model_str: str,
                 model_params: Dict[str, float],
                 num_coordinate_ascent_steps: int,
                 max_num_features: int = None,
                 plot_dir: str = None):

        self.model_str = model_str
        self.model_params = model_params
        assert num_coordinate_ascent_steps > 0
        self.num_coordinate_ascent_steps = num_coordinate_ascent_steps
        self.max_num_features = max_num_features
        self.plot_dir = plot_dir
        self.num_obs = None
        self.obs_dim = None

    def fit(self,
            observations: np.ndarray):

        num_obs, obs_dim = observations.shape
        self.num_obs = num_obs
        self.obs_dim = obs_dim
        if self.max_num_features is None:
            self.max_num_features = compute_max_num_features(
                alpha=self.model_params['alpha'],
                beta=self.model_params['beta'],
                num_obs=num_obs)

        offline_model = utils.inference_widjaja.OfflineFinite(
            obs_dim=obs_dim,
            num_obs=num_obs,
            max_num_features=self.max_num_features,
            alpha=self.model_params['alpha'],
            beta=self.model_params['beta'],
            sigma_a=np.sqrt(self.model_params['gaussian_prior_cov_scaling']),
            sigma_x=np.sqrt(self.model_params['gaussian_likelihood_cov_scaling']),
            t0=self.model_params['t0'],
            kappa=self.model_params['kappa'])

        offline_strategy = utils.inference_widjaja.Static(
            offline_model,
            observations,
            minibatch_size=num_obs,  # full batch
        )

        dish_eating_priors = np.zeros(shape=(num_obs, self.max_num_features))
        dish_eating_posteriors = np.zeros(shape=(num_obs, self.max_num_features))
        beta_param_1 = np.zeros(shape=(self.max_num_features,))
        beta_param_2 = np.zeros(shape=(self.max_num_features,))

        # Always use full batch
        obs_indices = slice(0, num_obs, 1)
        for step_idx in range(self.num_coordinate_ascent_steps):
            step_results = offline_strategy.step(obs_indices=obs_indices)
        dish_eating_priors[:, :] = step_results['dish_eating_prior']
        dish_eating_posteriors[:, :] = step_results['dish_eating_posterior']
        beta_param_1[:] = step_results['beta_param_1']
        beta_param_2[:] = step_results['beta_param_2']

        # shape (max number of features, obs dim)
        A_means = step_results['A_mean']
        # shape (max number of features, obs dim)
        # They assume a diagonal covariance. We will later expand.
        A_covs = step_results['A_cov']

        # Their model assumes a diagonal covariance. Convert to full covariance.
        A_covs = np.apply_along_axis(
            func1d=np.diag,
            axis=1,
            arr=A_covs,
        )
        variational_params = dict(
            A=dict(mean=A_means, cov=A_covs),
            pi=dict(param_1=beta_param_1, param_2=beta_param_2)
        )

        dish_eating_posteriors_running_sum = np.cumsum(dish_eating_posteriors, axis=0)

        num_dishes_poisson_rate_posteriors = np.sum(
            dish_eating_posteriors_running_sum > 1e-10,
            axis=1).reshape(-1, 1)

        num_dishes_poisson_rate_priors = np.full(fill_value=np.nan,
                                                 shape=num_dishes_poisson_rate_posteriors.shape)

        self.fit_results = dict(
            dish_eating_priors=dish_eating_priors,
            dish_eating_posteriors=dish_eating_posteriors,
            dish_eating_posteriors_running_sum=dish_eating_posteriors_running_sum,
            num_dishes_poisson_rate_priors=num_dishes_poisson_rate_priors,
            num_dishes_poisson_rate_posteriors=num_dishes_poisson_rate_posteriors,
            variational_params=variational_params,
            model_params=self.model_params,
        )

        return self.fit_results

    def sample_params_for_predictive_posterior(self,
                                               num_samples: int) -> Dict[str, np.ndarray]:

        var_params = self.fit_results['variational_params']
        max_num_features = var_params['pi']['param_1'].shape[0]
        indicators_probs = np.random.beta(
            a=var_params['pi']['param_1'][:],
            b=var_params['pi']['param_1'][:],
            size=(num_samples, max_num_features))
        features = np.array([np.random.multivariate_normal(mean=var_params['A']['mean'][k, :],
                                                           cov=var_params['A']['cov'][k, :],
                                                           size=num_samples)
                             for k in range(max_num_features)])
        # shape (num samples, num obs, obs dim)
        features = features.transpose(1, 0, 2)

        sampled_params = dict(
            indicators_probs=indicators_probs,
            features=features)

        return sampled_params

    def features_by_obs(self) -> np.ndarray:
        avg_feature = self.fit_results['variational_params']['A']['mean'][:]
        repeated_avg_feature = np.repeat(
            avg_feature[np.newaxis],
            repeats=self.num_obs,
            axis=0)
        return repeated_avg_feature

    def features_after_last_obs(self) -> np.ndarray:
        return self.fit_results['variational_params']['A']['mean'][:]


class HMCGibbsLinearGaussian(LinearGaussianModel):

    def __init__(self,
                 model_str: str,
                 model_params: Dict[str, float],
                 num_samples: int,
                 num_warmup_samples: int,
                 num_thinning_samples: int,
                 max_num_features: int = None):

        assert model_str in {'linear_gaussian', 'factor_analysis',
                             'nonnegative_matrix_factorization'}
        assert 'alpha' in model_params
        assert 'beta' in model_params
        assert model_params['alpha'] > 0
        assert model_params['beta'] > 0

        self.model_str = model_str
        self.model_params = model_params
        self.num_samples = num_samples
        self.max_num_features = max_num_features
        self.num_warmup_samples = num_warmup_samples
        self.num_thinning_samples = num_thinning_samples
        self.generative_model = None
        self.fit_results = None
        self.num_obs = None
        self.obs_dim = None

    def fit(self,
            observations: np.ndarray):

        self.num_obs, self.obs_dim = observations.shape
        if self.max_num_features is None:
            self.max_num_features = compute_max_num_features(
                alpha=self.model_params['alpha'],
                beta=self.model_params['beta'],
                num_obs=self.num_obs)

        self.generative_model = utils.numpyro_models.create_factor_analysis_model(
            model_params=self.model_params,
            num_obs=self.num_obs,
            max_num_features=self.max_num_features,
            obs_dim=self.obs_dim)

        hmc_kernel = numpyro.infer.NUTS(self.generative_model)
        kernel = numpyro.infer.DiscreteHMCGibbs(
            inner_kernel=hmc_kernel)
        mcmc = numpyro.infer.MCMC(
            kernel,
            num_warmup=self.num_warmup_samples,
            num_samples=self.num_samples,
            progress_bar=True)
        mcmc.run(jax.random.PRNGKey(0), obs=observations)
        # mcmc.print_summary()
        samples = mcmc.get_samples()

        # For some reason, Pyro puts the obs dimension last, so we transpose
        Z_samples = np.array(samples['Z']).transpose(0, 2, 1)
        dish_eating_posteriors = np.mean(
            Z_samples[::self.num_thinning_samples],
            axis=0)
        dish_eating_priors = np.full_like(
            dish_eating_posteriors,
            fill_value=np.nan)
        dish_eating_posteriors_running_sum = np.cumsum(dish_eating_posteriors, axis=0)
        num_dishes_poisson_rate_posteriors = np.sum(dish_eating_posteriors_running_sum > 1e-10,
                                                    axis=1).reshape(-1, 1)
        num_dishes_poisson_rate_priors = np.full(fill_value=np.nan,
                                                 shape=num_dishes_poisson_rate_posteriors.shape)

        samples = dict(
            v=dict(value=np.array(samples['sticks'][::self.num_thinning_samples, :])),
            A=dict(value=np.array(samples['A'][::self.num_thinning_samples, :, :])))

        self.fit_results = dict(
            dish_eating_priors=dish_eating_priors,
            dish_eating_posteriors=dish_eating_posteriors,
            dish_eating_posteriors_running_sum=dish_eating_posteriors_running_sum,
            num_dishes_poisson_rate_priors=num_dishes_poisson_rate_priors,
            num_dishes_poisson_rate_posteriors=num_dishes_poisson_rate_posteriors,
            samples=samples,
            model_params=self.model_params,
        )

        return self.fit_results

    def sample_params_for_predictive_posterior(self,
                                               num_samples: int) -> Dict[str, np.ndarray]:

        if self.fit_results is None:
            raise ValueError('Must call .fit() before calling .predict()')

        samples = self.fit_results['samples']

        random_mcmc_sample_idx = np.random.choice(
            np.arange(samples['v']['value'].shape[0]),
            size=num_samples,
            replace=True)
        indicators_probs = samples['v']['value'][random_mcmc_sample_idx, :]
        features = samples['A']['value'][random_mcmc_sample_idx, :]

        sampled_params = dict(
            indicators_probs=indicators_probs,  # shape (num samples, max num features)
            features=features,  # shape (num samples, max num features, obs dim)
        )

        return sampled_params

    def features_after_last_obs(self) -> np.ndarray:
        return np.mean(
            self.fit_results['samples']['A']['value'],
            axis=0)

    def features_by_obs(self) -> np.ndarray:
        avg_feature = np.mean(
            self.fit_results['samples']['A']['value'],
            axis=0)
        repeated_avg_feature = np.repeat(
            avg_feature[np.newaxis],
            repeats=self.num_obs,
            axis=0)
        return repeated_avg_feature
        

class RecursiveIBPLinearGaussian(LinearGaussianModel):

    def __init__(self,
                 model_str: str,
                 model_params: Dict[str, float],
                 ossify_features: bool = True,
                 numerically_optimize: bool = False,
                 learning_rate: float = 1e0,
                 coord_ascent_update_type: str = 'sequential',
                 num_coord_ascent_steps_per_obs: int = 3,
                 plot_dir: str = None):

        # Check validity of input params
        assert model_str in {'linear_gaussian', 'factor_analysis',
                             'nonnegative_matrix_factorization'}
        assert 'alpha' in model_params
        assert 'beta' in model_params
        assert model_params['alpha'] > 0
        assert model_params['beta'] > 0
        assert coord_ascent_update_type in {'simultaneous', 'sequential'}
        if numerically_optimize:
            assert learning_rate is not None
            assert learning_rate > 0.

        if numerically_optimize is False:
            learning_rate = np.nan
        else:
            assert isinstance(learning_rate, float)

        self.model_str = model_str
        self.model_params = model_params
        self.plot_dir = plot_dir
        self.ossify_features = ossify_features
        self.coord_ascent_update_type = coord_ascent_update_type
        self.num_coord_ascent_steps_per_obs = num_coord_ascent_steps_per_obs
        self.numerically_optimize = numerically_optimize
        self.learning_rate = learning_rate
        self.variational_params = None

    def fit(self,
            observations):
        """
        Perform inference using Recursive IBP with specified likelihood.
        """

        num_obs, obs_dim = observations.shape

        # Note: the expected number of latents grows logarithmically as a*b*log(1 + N/beta)
        # The 10 is a hopefully conservative heuristic to preallocate.
        max_num_features = compute_max_num_features(
            alpha=self.model_params['alpha'],
            beta=self.model_params['beta'],
            num_obs=num_obs)

        # The recursion does not require recording the full history of priors/posteriors
        # but we record the full history for subsequent analysis.
        dish_eating_priors = torch.zeros(
            (num_obs + 1, max_num_features),  # Add 1 to the number of observations to use 1-based indexing
            dtype=torch.float32,
            requires_grad=self.numerically_optimize)

        dish_eating_posteriors = torch.zeros(
            (num_obs + 1, max_num_features),
            dtype=torch.float32,
            requires_grad=self.numerically_optimize)

        dish_eating_posteriors_running_sum = torch.zeros(
            (num_obs + 1, max_num_features),
            dtype=torch.float32,
            requires_grad=self.numerically_optimize)

        non_eaten_dishes_posteriors_running_prod = torch.ones(
            (num_obs + 1, max_num_features),
            dtype=torch.float32,
            requires_grad=self.numerically_optimize)

        num_dishes_poisson_rate_priors = torch.zeros(
            (num_obs + 1, 1),
            dtype=torch.float32,
            requires_grad=self.numerically_optimize)

        num_dishes_poisson_rate_posteriors = torch.zeros(
            (num_obs + 1, 1),
            dtype=torch.float32,
            requires_grad=self.numerically_optimize)

        # we use half covariance because we want to numerically optimize
        A_half_covs = torch.stack([
            np.sqrt(self.model_params['gaussian_prior_cov_scaling']) * torch.eye(obs_dim).float()
            for _ in range((num_obs + 1) * max_num_features)])
        A_half_covs = A_half_covs.view(num_obs + 1, max_num_features, obs_dim, obs_dim)

        # dict mapping variables to variational params
        self.variational_params = dict(
            Z=dict(  # variational params for binary indicators
                prob=torch.full(
                    size=(num_obs + 1, max_num_features),
                    fill_value=np.nan,
                    dtype=torch.float32),
            ),
            A=dict(  # variational params for Gaussian features
                mean=torch.full(
                    size=(num_obs + 1, max_num_features, obs_dim),
                    fill_value=0.,
                    dtype=torch.float32),
                # mean=A_mean,
                half_cov=A_half_covs),
        )

        # If we are optimizing numerically, set requires gradient to true, otherwise false
        for var_name, var_param_dict in self.variational_params.items():
            for param_name, param_tensor in var_param_dict.items():
                param_tensor.requires_grad = self.numerically_optimize

        torch_observations = torch.from_numpy(observations).float()
        latent_indices = np.arange(max_num_features)

        # before the first observation, there are exactly 0 dishes
        num_dishes_poisson_rate_posteriors[0, 0] = 0.

        # REMEMBER: we added +1 to all the record-keeping arrays. Starting with 1
        # makes indexing consistent with the paper notation.
        for obs_idx, torch_observation in enumerate(torch_observations[:num_obs], start=1):

            # construct number of dishes Poisson rate prior
            num_dishes_poisson_rate_priors[obs_idx, :] = num_dishes_poisson_rate_posteriors[obs_idx - 1, :]
            num_dishes_poisson_rate_priors[obs_idx, :] += self.model_params['alpha'] * self.model_params['beta'] \
                                                          / (self.model_params['beta'] + obs_idx - 1)
            # Recursion: 1st term
            dish_eating_prior = torch.clone(
                dish_eating_posteriors_running_sum[obs_idx - 1, :]) / (self.model_params['beta'] + obs_idx - 1)
            # Recursion: 2nd term; don't subtract 1 from latent indices b/c zero based indexing
            dish_eating_prior += poisson.cdf(k=latent_indices, mu=num_dishes_poisson_rate_posteriors[obs_idx - 1, :])
            # Recursion: 3rd term; don't subtract 1 from latent indices b/c zero based indexing
            dish_eating_prior -= poisson.cdf(k=latent_indices, mu=num_dishes_poisson_rate_priors[obs_idx, :])

            # record latent prior
            dish_eating_priors[obs_idx, :] = dish_eating_prior.clone()

            # initialize dish eating posterior to dish eating prior, before beginning inference
            self.variational_params['Z']['prob'].data[obs_idx, :] = dish_eating_prior.clone()
            dish_eating_posteriors.data[obs_idx, :] = dish_eating_prior.clone()

            # initialize features to previous features as starting point for optimization
            # Use .data to not break backprop
            for param_name, param_tensor in self.variational_params['A'].items():
                param_tensor.data[obs_idx, :] = param_tensor.data[obs_idx - 1, :]

            if self.plot_dir is not None:
                num_cols = 4
                fig, axes = plt.subplots(
                    nrows=self.num_coord_ascent_steps_per_obs,
                    ncols=num_cols,
                    # sharex=True,
                    # sharey=True,
                    figsize=(num_cols * 4, self.num_coord_ascent_steps_per_obs * 4))

            approx_lower_bounds = []
            for vi_idx in range(self.num_coord_ascent_steps_per_obs):
                if self.numerically_optimize:
                    # TODO: untested!
                    # maximize approximate lower bound with respect to A's params
                    approx_lower_bound = recursive_ibp_compute_approx_lower_bound(
                        torch_observation=torch_observation,
                        obs_idx=obs_idx,
                        dish_eating_prior=dish_eating_prior,
                        variational_params=self.variational_params)
                    approx_lower_bounds.append(approx_lower_bound.item())
                    approx_lower_bound.backward()

                    # scale learning rate by posterior(A_k) / sum_n prev_posteriors(A_k)
                    # scale by 1/num_vi_steps so that after num_vi_steps, we've moved O(1)
                    scaled_learning_rate = self.learning_rate * torch.divide(
                        dish_eating_posteriors[obs_idx, :],
                        dish_eating_posteriors[obs_idx, :] + dish_eating_posteriors_running_sum[obs_idx - 1, :])
                    scaled_learning_rate /= self.num_coord_ascent_steps_per_obs
                    scaled_learning_rate[torch.isnan(scaled_learning_rate)] = 0.
                    scaled_learning_rate[torch.isinf(scaled_learning_rate)] = 0.

                    # make sure no gradient when applying gradient updates
                    with torch.no_grad():
                        for var_name, var_dict in self.variational_params.items():
                            for param_name, param_tensor in var_dict.items():
                                # the scaled learning rate has shape (num latents,) aka (num obs,)
                                # we need to create extra dimensions of size 1 for broadcasting to work
                                # because param_tensor can have variable number of dimensions e.g. (num obs, obs dim) for mean
                                # or (num obs, obs dim, obs dim) for covariance, we need to dynamically
                                # add the correct number of dimensions
                                # Also, exclude dimension 0 because that's for the observation index
                                reshaped_scaled_learning_rate = scaled_learning_rate.view(
                                    [param_tensor.shape[1]] + [1 for _ in range(len(param_tensor.shape[2:]))])
                                if param_tensor.grad is None:
                                    continue
                                else:
                                    scaled_param_tensor_grad = torch.multiply(
                                        reshaped_scaled_learning_rate,
                                        param_tensor.grad[obs_idx, :])
                                    param_tensor.data[obs_idx, :] += scaled_param_tensor_grad
                                    utils.torch_helpers.assert_no_nan_no_inf(param_tensor.data[:obs_idx + 1])

                                    # zero gradient manually
                                    param_tensor.grad = None

                elif not self.numerically_optimize:
                    with torch.no_grad():
                        logging.info(f'Obs Idx: {obs_idx}, VI idx: {vi_idx}')

                        # tracemalloc.start()
                        # start_time = timer()
                        # approx_lower_bound = recursive_ibp_compute_approx_lower_bound(
                        #     torch_observation=torch_observation,
                        #     obs_idx=obs_idx,
                        #     dish_eating_prior=dish_eating_prior,
                        #     variational_params=self.variational_params,
                        #     sigma_obs_squared=self.model_params['gaussian_likelihood_cov_scaling'])
                        # approx_lower_bounds.append(approx_lower_bound.item())
                        # stop_time = timer()
                        # current, peak = tracemalloc.get_traced_memory()
                        # logging.debug(f"memory:recursive_ibp_compute_approx_lower_bound:"
                        #               f"current={current / 10 ** 6}MB; peak={peak / 10 ** 6}MB")
                        # logging.debug(f'runtime:recursive_ibp_compute_approx_lower_bound: {stop_time - start_time}')
                        # tracemalloc.stop()
                        # logging.info(prof.key_averages().table(sort_by="self_cpu_memory_usage"))

                        # tracemalloc.start()
                        # start_time = timer()
                        A_means, A_half_covs = recursive_ibp_optimize_gaussian_params(
                            torch_observation=torch_observation,
                            obs_idx=obs_idx,
                            vi_idx=vi_idx,
                            variable_variational_params=self.variational_params,
                            simultaneous_or_sequential=self.coord_ascent_update_type,
                            sigma_obs_squared=self.model_params['gaussian_likelihood_cov_scaling'])
                        # stop_time = timer()
                        # current, peak = tracemalloc.get_traced_memory()
                        # logging.debug(f"memory:recursive_ibp_optimize_gaussian_params:"
                        #               f"current={current / 10 ** 6}MB; peak={peak / 10 ** 6}MB")
                        # logging.debug(f'runtime:recursive_ibp_optimize_gaussian_params: {stop_time - start_time}')
                        # tracemalloc.stop()

                        if self.ossify_features:
                            # TODO: refactor into own function
                            normalizing_const = torch.add(
                                self.variational_params['Z']['prob'].data[obs_idx, :],
                                dish_eating_posteriors_running_sum[obs_idx - 1, :])
                            history_weighted_A_means = torch.add(
                                torch.multiply(self.variational_params['Z']['prob'].data[obs_idx, :, None],
                                               A_means),
                                torch.multiply(dish_eating_posteriors_running_sum[obs_idx - 1, :, None],
                                               self.variational_params['A']['mean'].data[obs_idx - 1, :]))
                            history_weighted_A_half_covs = torch.add(
                                torch.multiply(self.variational_params['Z']['prob'].data[obs_idx, :, None, None],
                                               A_half_covs),
                                torch.multiply(dish_eating_posteriors_running_sum[obs_idx - 1, :, None, None],
                                               self.variational_params['A']['half_cov'].data[obs_idx - 1, :]))
                            A_means = torch.divide(history_weighted_A_means, normalizing_const[:, None])
                            A_half_covs = torch.divide(history_weighted_A_half_covs, normalizing_const[:, None, None])
                            # if cumulative probability mass is 0, we compute 0/0 and get NaN. Need to mask
                            A_means[normalizing_const == 0.] = \
                                self.variational_params['A']['mean'][obs_idx - 1][normalizing_const == 0.]
                            A_half_covs[normalizing_const == 0.] = \
                                self.variational_params['A']['half_cov'][obs_idx - 1][normalizing_const == 0.]

                            utils.torch_helpers.assert_no_nan_no_inf(A_means)
                            utils.torch_helpers.assert_no_nan_no_inf(A_half_covs)

                        self.variational_params['A']['mean'].data[obs_idx, :] = A_means
                        self.variational_params['A']['half_cov'].data[obs_idx, :] = A_half_covs

                        # maximize approximate lower bound with respect to Z's params
                        # tracemalloc.start()
                        # start_time = timer()
                        Z_probs = recursive_ibp_optimize_bernoulli_params(
                            torch_observation=torch_observation,
                            obs_idx=obs_idx,
                            vi_idx=vi_idx,
                            dish_eating_prior=dish_eating_prior,
                            variational_params=self.variational_params,
                            simultaneous_or_sequential=self.coord_ascent_update_type,
                            sigma_obs_squared=self.model_params['gaussian_likelihood_cov_scaling'])
                        # stop_time = timer()
                        # current, peak = tracemalloc.get_traced_memory()
                        # logging.debug(f"memory:recursive_ibp_optimize_bernoulli_params:"
                        #               f"current={current / 10 ** 6}MB; peak={peak / 10 ** 6}MB")
                        # logging.debug(f'runtime:recursive_ibp_optimize_bernoulli_params: {stop_time - start_time}')
                        # tracemalloc.stop()

                        self.variational_params['Z']['prob'].data[obs_idx, :] = Z_probs

                        # record dish-eating posterior
                        dish_eating_posteriors.data[obs_idx, :] = \
                            self.variational_params['Z']['prob'][obs_idx, :].clone()

                else:
                    raise ValueError(f'Impermissible value of numerically_optimize: {self.numerically_optimize}')

                if self.plot_dir is not None:
                    fig.suptitle(f'Obs: {obs_idx}, {self.coord_ascent_update_type}')
                    axes[vi_idx, 0].set_ylabel(f'VI Step: {vi_idx + 1}')
                    axes[vi_idx, 0].set_title('Individual Features')
                    axes[vi_idx, 0].scatter(observations[:obs_idx, 0],
                                            observations[:obs_idx, 1],
                                            # s=3,
                                            color='k',
                                            label='Observations')
                    for feature_idx in range(10):
                        axes[vi_idx, 0].plot(
                            [0, self.variational_params['A']['mean'][obs_idx, feature_idx, 0].item()],
                            [0, self.variational_params['A']['mean'][obs_idx, feature_idx, 1].item()],
                            label=f'{feature_idx}')
                    # axes[0].legend()

                    axes[vi_idx, 1].set_title('Inferred Indicator Probabilities')
                    # axes[vi_idx, 1].set_xlabel('Indicator Index')
                    axes[vi_idx, 1].scatter(
                        1 + np.arange(10),
                        dish_eating_priors[obs_idx, :10].detach().numpy(),
                        label='Prior')
                    axes[vi_idx, 1].scatter(
                        1 + np.arange(10),
                        dish_eating_posteriors[obs_idx, :10].detach().numpy(),
                        label='Posterior')
                    axes[vi_idx, 1].legend()

                    weighted_features = np.multiply(
                        self.variational_params['A']['mean'][obs_idx, :, :].detach().numpy(),
                        dish_eating_posteriors[obs_idx].unsqueeze(1).detach().numpy(),
                    )
                    axes[vi_idx, 2].set_title('Weighted Features')
                    axes[vi_idx, 2].scatter(observations[:obs_idx, 0],
                                            observations[:obs_idx, 1],
                                            # s=3,
                                            color='k',
                                            label='Observations')
                    for feature_idx in range(10):
                        axes[vi_idx, 2].plot([0, weighted_features[feature_idx, 0]],
                                             [0, weighted_features[feature_idx, 1]],
                                             label=f'{feature_idx}',
                                             zorder=feature_idx + 1,
                                             # alpha=dish_eating_posteriors[obs_idx, feature_idx].item(),
                                             )

                    cumulative_weighted_features = np.cumsum(weighted_features, axis=0)
                    axes[vi_idx, 3].set_title('Cumulative Weighted Features')
                    axes[vi_idx, 3].scatter(observations[:obs_idx, 0],
                                            observations[:obs_idx, 1],
                                            # s=3,
                                            color='k',
                                            label='Observations')
                    for feature_idx in range(10):
                        axes[vi_idx, 3].plot(
                            [0 if feature_idx == 0 else cumulative_weighted_features[feature_idx - 1, 0],
                             cumulative_weighted_features[feature_idx, 0]],
                            [0 if feature_idx == 0 else cumulative_weighted_features[feature_idx - 1, 1],
                             cumulative_weighted_features[feature_idx, 1]],
                            label=f'{feature_idx}',
                            alpha=dish_eating_posteriors[obs_idx, feature_idx].item())

                with torch.no_grad():

                    # record dish-eating posterior
                    dish_eating_posteriors.data[obs_idx, :] = self.variational_params['Z']['prob'][obs_idx, :].clone()

                    # update running sum of posteriors
                    dish_eating_posteriors_running_sum[obs_idx, :] = torch.add(
                        dish_eating_posteriors_running_sum[obs_idx - 1, :],
                        dish_eating_posteriors[obs_idx, :])

                    # update how many dishes have been sampled
                    non_eaten_dishes_posteriors_running_prod[obs_idx, :] = np.multiply(
                        non_eaten_dishes_posteriors_running_prod[obs_idx - 1, :],
                        1. - dish_eating_posteriors[obs_idx, :],
                        # p(z_{tk} = 0|o_{\leq t}) = 1 - p(z_{tk} = 1|o_{\leq t})
                    )

                    # approx_lower_bound = recursive_ibp_compute_approx_lower_bound(
                    #     torch_observation=torch_observation,
                    #     obs_idx=obs_idx,
                    #     dish_eating_prior=dish_eating_prior,
                    #     variational_params=self.variational_params,
                    #     sigma_obs_squared=self.model_params['gaussian_likelihood_cov_scaling'])
                    # approx_lower_bounds.append(approx_lower_bound.item())

                    num_dishes_poisson_rate_posteriors[obs_idx] = torch.sum(
                        1. - non_eaten_dishes_posteriors_running_prod[obs_idx, :])

                    # update running sum of which customers ate which dishes
                    dish_eating_posteriors_running_sum[obs_idx] = torch.add(
                        dish_eating_posteriors_running_sum[obs_idx - 1, :],
                        dish_eating_posteriors[obs_idx, :])

            if self.plot_dir is not None:
                plt.savefig(os.path.join(self.plot_dir,
                                         f'{self.coord_ascent_update_type}_params_obs={obs_idx}.png'),
                            bbox_inches='tight',
                            dpi=300)
                # plt.show()
                plt.close()

                plt.scatter(1 + np.arange(len(approx_lower_bounds)),
                            approx_lower_bounds)
                plt.xlabel('VI Step')
                plt.ylabel('VI Approx Lower Bound')
                plt.savefig(os.path.join(self.plot_dir,
                                         f'{self.coord_ascent_update_type}_approxlowerbound_obs={obs_idx}.png'),
                            bbox_inches='tight',
                            dpi=300)
                # plt.show()
                plt.close()

        # Remember to cut off the first index.y
        numpy_variable_params = {
            var_name: {param_name: param_tensor.detach().numpy()[1:]
                       for param_name, param_tensor in var_params.items()}
            for var_name, var_params in self.variational_params.items()
        }
        self.fit_results = dict(
            dish_eating_priors=dish_eating_priors.numpy()[1:],
            dish_eating_posteriors=dish_eating_posteriors.numpy()[1:],
            dish_eating_posteriors_running_sum=dish_eating_posteriors_running_sum.numpy()[1:],
            non_eaten_dishes_posteriors_running_prod=non_eaten_dishes_posteriors_running_prod.numpy()[1:],
            num_dishes_poisson_rate_priors=num_dishes_poisson_rate_priors.numpy()[1:],
            num_dishes_poisson_rate_posteriors=num_dishes_poisson_rate_posteriors.numpy()[1:],
            variational_params=numpy_variable_params,
            model_params=self.model_params,
        )

        return self.fit_results

    def sample_params_for_predictive_posterior(self,
                                               num_samples: int) -> Dict[str, np.ndarray]:

        # obs index is customer index (1-based)
        # add one because we are predicting the next customer
        obs_idx = self.fit_results['dish_eating_posteriors'].shape[0] + 1
        max_num_features = self.fit_results['variational_params']['Z']['prob'].shape[1]
        latent_indices = np.arange(max_num_features)

        # Construct prior over next time step
        num_dishes_poisson_rate_prior = self.fit_results['num_dishes_poisson_rate_posteriors'][obs_idx - 2, :]
        num_dishes_poisson_rate_prior += self.model_params['alpha'] * self.model_params['beta'] \
                                         / (self.model_params['beta'] + obs_idx - 1)

        dish_eating_prior = self.fit_results['dish_eating_posteriors_running_sum'][obs_idx - 2, :] \
                            / (self.model_params['beta'] + obs_idx - 1)
        # Recursion: 2nd term; don't subtract 1 from latent indices b/c zero based indexing
        dish_eating_prior += poisson.cdf(
            k=latent_indices,
            mu=self.fit_results['num_dishes_poisson_rate_posteriors'][obs_idx - 2, :])
        # Recursion: 3rd term; don't subtract 1 from latent indices b/c zero based indexing

        dish_eating_prior -= poisson.cdf(
            k=latent_indices,
            mu=num_dishes_poisson_rate_prior)

        # TODO: Investigate why we get negative values; difference of CDFs, perhaps?
        # set floating errors to small values
        dish_eating_prior[dish_eating_prior < 1e-10] = 1e-10

        indicators_probs = np.random.binomial(
            n=1,
            p=dish_eating_prior[np.newaxis, :],
            size=(num_samples, max_num_features))

        var_params = self.fit_results['variational_params']

        covs = utils.numpy_helpers.convert_half_cov_to_cov(
            half_cov=var_params['A']['half_cov'][-1, :, :, :])
        features = np.stack([np.random.multivariate_normal(mean=var_params['A']['mean'][-1, k, :],
                                                           cov=covs[k],
                                                           size=num_samples)
                             for k in range(max_num_features)])
        # shape = (num samples, max num features, obs dim)
        features = features.transpose(1, 0, 2)

        sampled_params = dict(
            indicators_probs=indicators_probs,
            features=features)

        return sampled_params

    def features_after_last_obs(self) -> np.ndarray:
        return self.fit_results['variational_params']['A']['mean'][-1, :, :]

    def features_by_obs(self) -> np.ndarray:
        return self.fit_results['variational_params']['A']['mean'][:, :, :]


class WidjajaLinearGaussian(LinearGaussianModel):
    """
    Implementation of Widjaja 2017 Streaming Variational Inference for the IBP.
    """

    def __init__(self,
                 model_str: str,
                 model_params: Dict[str, float],
                 max_num_features: int = None,
                 plot_dir: str = None
                 ):
        self.model_str = model_str
        self.model_params = model_params
        self.max_num_features = max_num_features
        self.plot_dir = plot_dir

    def fit(self,
            observations: np.ndarray):
        num_obs, obs_dim = observations.shape
        if self.max_num_features is None:
            self.max_num_features = compute_max_num_features(
                alpha=self.model_params['alpha'],
                beta=self.model_params['beta'],
                num_obs=num_obs)

        online_model = utils.inference_widjaja.OnlineFinite(
            obs_dim=obs_dim,
            max_num_features=self.max_num_features,
            alpha=self.model_params['alpha'],
            beta=self.model_params['beta'],
            sigma_a=np.sqrt(self.model_params['gaussian_prior_cov_scaling']),
            sigma_x=np.sqrt(self.model_params['gaussian_likelihood_cov_scaling']),
            t0=1,
            kappa=0.5)

        online_strategy = utils.inference_widjaja.Static(
            online_model,
            observations,
            minibatch_size=10)

        dish_eating_priors = np.zeros(shape=(num_obs, self.max_num_features))

        dish_eating_posteriors = np.zeros(shape=(num_obs, self.max_num_features))

        beta_param_1 = np.zeros(shape=(num_obs, self.max_num_features))
        beta_param_2 = np.zeros(shape=(num_obs, self.max_num_features))

        A_means = np.zeros(shape=(num_obs, self.max_num_features, obs_dim))
        # They assume a diagonal covariance. We will later expand.
        A_covs = np.zeros(shape=(num_obs, self.max_num_features, obs_dim))

        for obs_idx in range(num_obs):
            obs_indices = slice(obs_idx, obs_idx + 1, 1)
            step_results = online_strategy.step(obs_indices=obs_indices)
            dish_eating_priors[obs_idx, :] = step_results['dish_eating_prior'][0, :]
            dish_eating_posteriors[obs_idx] = step_results['dish_eating_posterior'][0, :]
            A_means[obs_idx] = step_results['A_mean']
            A_covs[obs_idx] = step_results['A_cov']
            beta_param_1[obs_idx] = step_results['beta_param_1']
            beta_param_2[obs_idx] = step_results['beta_param_2']

        # Their model assumes a diagonal covariance. Convert to full covariance.
        A_covs = np.apply_along_axis(
            func1d=np.diag,
            axis=2,
            arr=A_covs,
        )
        variational_params = dict(
            A=dict(mean=A_means, cov=A_covs),
            pi=dict(param_1=beta_param_1, param_2=beta_param_2)
        )

        dish_eating_posteriors_running_sum = np.cumsum(dish_eating_posteriors, axis=0)

        num_dishes_poisson_rate_posteriors = np.sum(dish_eating_posteriors_running_sum > 1e-2,
                                                    axis=1).reshape(-1, 1)

        dish_eating_priors_running_sum = np.cumsum(dish_eating_priors, axis=0)
        num_dishes_poisson_rate_priors = np.sum(dish_eating_priors_running_sum > 1e-2,
                                                axis=1).reshape(-1, 1)

        self.fit_results = dict(
            dish_eating_priors=dish_eating_priors,
            dish_eating_posteriors=dish_eating_posteriors,
            dish_eating_posteriors_running_sum=dish_eating_posteriors_running_sum,
            num_dishes_poisson_rate_priors=num_dishes_poisson_rate_priors,
            num_dishes_poisson_rate_posteriors=num_dishes_poisson_rate_posteriors,
            variational_params=variational_params,
            model_params=self.model_params,
        )

        return self.fit_results

    def sample_params_for_predictive_posterior(self,
                                               num_samples: int) -> Dict[str, np.ndarray]:

        var_params = self.fit_results['variational_params']
        # TODO: investigate why some param_2 are negative and how to stop it.
        # Something in the original Widjaja code is screwing up.
        param_1 = var_params['pi']['param_1']
        param_1[param_1 < 1e-10] = 1e-10
        param_2 = var_params['pi']['param_2']
        param_2[param_2 < 1e-10] = 1e-10

        max_num_features = param_1.shape[1]
        # shape (num samples, num features)
        indicators_probs = np.random.beta(
            a=param_1[np.newaxis, -1, :],
            b=param_2[np.newaxis, -1, :],
            size=(num_samples, max_num_features))
        features = np.stack([np.random.multivariate_normal(mean=var_params['A']['mean'][-1, k, :],
                                                           cov=var_params['A']['cov'][-1, k, :],
                                                           size=num_samples)
                             for k in range(max_num_features)])
        # shape = (num samples, max num features, obs dim)
        features = features.transpose(1, 0, 2)

        sampled_params = dict(
            indicators_probs=indicators_probs,
            features=features)

        return sampled_params

    def features_after_last_obs(self) -> np.ndarray:
        return self.fit_results['variational_params']['A']['mean'][-1, :, :]

    def features_by_obs(self) -> np.ndarray:
        return self.fit_results['variational_params']['A']['mean'][:, :, :]


def compute_max_num_features(alpha: float,
                             beta: float,
                             num_obs: int,
                             prefactor: int = 10):
    # Note: the expected number of latents grows logarithmically as a*b*log(1 + N/sticks)
    # The 10 is a hopefully conservative heuristic to preallocate.
    return prefactor * int(alpha * beta * np.log(1 + num_obs / beta))


def create_new_feature_params_multivariate_normal(torch_observation: torch.Tensor,
                                                  dish_eating_prior: torch.Tensor,
                                                  obs_idx: int,
                                                  likelihood_params: Dict[str, torch.Tensor],
                                                  sigma_obs_squared: int = 1.):
    # data is necessary to not break backprop
    # see https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
    utils.torch_helpers.assert_no_nan_no_inf(torch_observation)
    max_num_features = likelihood_params['means'].shape[0]
    obs_dim = torch_observation.shape[0]

    # subtract the contribution from existing likelihood params
    torch_residual = torch_observation - torch.matmul(dish_eating_prior, likelihood_params['means'])

    # Create new params by regressing prior on residuals with two additions:
    #   1. L1 regularization to encourage fewer new features are introduced.
    #   2. Divergence from
    cp_features_var = cp.Variable(shape=(max_num_features, obs_dim))
    cp_sse_fn = 0.5 * cp.sum_squares(
        torch_residual.detach().numpy() - cp.matmul(dish_eating_prior, cp_features_var))
    cp_l1_fn = cp.norm1(cp_features_var)
    # TODO: add Gaussian prior regularization
    cp_objective = cp.Minimize(cp_sse_fn + cp_l1_fn)
    prob = cp.Problem(objective=cp_objective)
    prob.solve()
    torch_features = torch.from_numpy(cp_features_var.value)

    # frequently, we get small floating point content e.g. 1e-35. These are numerical errors.
    torch_features[torch_features < 1e-10] = 0.

    # data is necessary to not break backprop
    # see https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
    likelihood_params['means'].data[:, :] = torch_features


def posterior_multivariate_normal_linear_regression_simultaneous(torch_observation,
                                                                 likelihood_params,
                                                                 dish_eating_prior):
    max_num_dishes = dish_eating_prior.shape[0]
    cp_dish_eating_var = cp.Variable(shape=(1, max_num_dishes))

    one_minus_Z_prior = 1. - dish_eating_prior.numpy()
    log_one_minus_Z_prior = np.log(one_minus_Z_prior)
    cp_sse_fn = 0.5 * cp.sum_squares(torch_observation - cp_dish_eating_var @ likelihood_params['means'])
    cp_prior_fn = cp.sum(
        cp.multiply(cp_dish_eating_var,
                    np.log(np.divide(dish_eating_prior, one_minus_Z_prior))) + log_one_minus_Z_prior)
    # cp_l1_fn = cp.norm1(cp_dish_eating_var)

    cp_constraints = [0 <= cp_dish_eating_var, cp_dish_eating_var <= 1]
    cp_objective = cp.Minimize(cp_sse_fn - cp_prior_fn)
    prob = cp.Problem(objective=cp_objective, constraints=cp_constraints)
    prob.solve()
    dish_eating_posterior = torch.from_numpy(cp_dish_eating_var.value)
    return dish_eating_posterior


def posterior_multivariate_normal_linear_regression_forward_stepwise(torch_observation,
                                                                     obs_idx,
                                                                     likelihood_params,
                                                                     dish_eating_prior):
    max_num_dishes = dish_eating_prior.shape[0]

    dish_eating_posterior = torch.zeros_like(dish_eating_prior)

    component_explained_by_earlier_features = torch.zeros_like(torch_observation)
    log_likelihoods_per_latent_equal_one = torch.zeros(max_num_dishes)
    log_likelihoods_per_latent_equal_zero = torch.zeros(max_num_dishes)

    import matplotlib.pyplot as plt

    for dish_idx in range(max_num_dishes):
        dish_mean = likelihood_params['mean'][dish_idx]

        torch_observation_minus_component_explained_by_earlier_features = \
            torch_observation - component_explained_by_earlier_features

        if obs_idx == 2:
            plt.scatter(component_explained_by_earlier_features[0],
                        component_explained_by_earlier_features[1],
                        label=r'$\sum_{k\prime < k} \phi_{k\prime} p(z_k)$')
            plt.scatter(component_explained_by_earlier_features[0] + dish_mean[0],
                        component_explained_by_earlier_features[1] + dish_mean[1],
                        label=r'$\sum_{k\prime < k} \phi_{k\prime} p(z_k) + \phi_k$')
            plt.scatter(torch_observation[0],
                        torch_observation[1],
                        label=r'$o_{2}$')
            plt.title(f'k={dish_idx + 1}')
            plt.legend()
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            plt.show()

        # compute p(o_t|z_{tk} = 1, z_{t, -k})
        mv_normal_latent_equal_one = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=dish_mean,
            covariance_matrix=likelihood_params['cov'])
        log_likelihoods_per_latent_equal_one[dish_idx] = mv_normal_latent_equal_one.log_prob(
            value=torch_observation_minus_component_explained_by_earlier_features)

        # compute p(o_t|z_{tk} = 0, z_{t, -k})
        mv_normal_latent_equal_zero = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros_like(dish_mean),
            covariance_matrix=likelihood_params['cov'])
        log_likelihoods_per_latent_equal_zero[dish_idx] = mv_normal_latent_equal_zero.log_prob(
            value=torch_observation_minus_component_explained_by_earlier_features)

        # log_ratio = log_likelihoods_per_latent_equal_one[dish_idx] - log_likelihoods_per_latent_equal_zero[dish_idx]

        # typically, we would compute the posterior as:
        # p(z=1|o, history) = p(z=1, o|history) / p(o|history)
        #                   = p(z=1, o |history) / (p(z=1, o |history) + p(z=0, o |history))
        # this is numerically unstable. Instead, we use the following likelihood ratio-based approach
        # p(z=1|o, history) / p(z=0|o, history) = p(o|z=1) p(z=1|history) / (p(o|z=0) p(z=0|history))
        # rearranging, we see that
        # p(z=1|o, history) / p(z=0|o, history) =
        #               exp(log p(o|z=1) + log p(z=1|history) - log (p(o|z=0) - log p(z=0|history)))
        # Noting that p(z=0|o, history) = 1 - p(z=1|o, history), if we define the RHS as A,
        # then p(z=1|o, history) = A / (1 + A)
        A_argument = log_likelihoods_per_latent_equal_one[dish_idx] \
                     - log_likelihoods_per_latent_equal_zero[dish_idx]
        A_argument += torch.log(dish_eating_prior[dish_idx]) \
                      - torch.log(1 - dish_eating_prior[dish_idx])
        A = torch.exp(A_argument)
        dish_eating_posterior[dish_idx] = torch.divide(A, 1 + A)

        # if A large/infinite, we want the result to approach 1.
        if torch.isinf(A):
            dish_eating_posterior[dish_idx] = 1.

        # update component explained by earlier features
        component_explained_by_earlier_features += dish_mean * dish_eating_posterior[dish_idx]

    return dish_eating_posterior


def posterior_multivariate_normal_linear_regression_leave_one_out(torch_observation,
                                                                  obs_idx,
                                                                  likelihood_params,
                                                                  dish_eating_prior):
    # TODO: figure out how to do gradient descent using the post-gradient step means
    max_num_dishes = dish_eating_prior.shape[0]

    dish_eating_posterior = torch.clone(dish_eating_prior)

    log_likelihoods_per_latent_equal_one = torch.zeros(max_num_dishes)
    log_likelihoods_per_latent_equal_zero = torch.zeros(max_num_dishes)
    mask = torch.ones(size=(max_num_dishes,), dtype=torch.bool)

    import matplotlib.pyplot as plt

    for iter_idx in range(3):
        for dish_idx in range(max_num_dishes):
            mask[dish_idx] = False
            dish_mean = likelihood_params['mean'][~mask]
            other_dish_means = likelihood_params['mean'][mask]
            component_explained_by_other_features = torch.sum(
                torch.multiply(dish_eating_posterior[mask].unsqueeze(1),
                               other_dish_means),
                axis=0)
            torch_observation_minus_component_explained_by_other_features = torch.subtract(
                torch_observation,
                component_explained_by_other_features)

            if obs_idx == 1 and dish_idx < 7:
                plt.scatter(component_explained_by_other_features[0],
                            component_explained_by_other_features[1],
                            label=r'$\sum_{k\prime < k} \phi_{k\prime} p(z_k)$')
                plt.scatter(component_explained_by_other_features[0] + dish_mean[0, 0],
                            component_explained_by_other_features[1] + dish_mean[0, 1],
                            label=r'$\sum_{k\prime < k} \phi_{k\prime} p(z_k) + \phi_k$')
                plt.scatter(torch_observation[0],
                            torch_observation[1],
                            label=r'$o_{2}$')
                plt.title(f'k={dish_idx + 1}')
                plt.legend()
                plt.xlim(-4, 4)
                plt.ylim(-4, 4)
                plt.show()

            # compute p(o_t|z_{tk} = 1, z_{t, -k})
            mv_normal_latent_equal_one = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=dish_mean,
                covariance_matrix=likelihood_params['cov'])
            log_likelihoods_per_latent_equal_one[dish_idx] = mv_normal_latent_equal_one.log_prob(
                value=torch_observation_minus_component_explained_by_other_features)

            # compute p(o_t|z_{tk} = 0, z_{t, -k})
            mv_normal_latent_equal_zero = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=torch.zeros_like(dish_mean),
                covariance_matrix=likelihood_params['cov'])
            log_likelihoods_per_latent_equal_zero[dish_idx] = mv_normal_latent_equal_zero.log_prob(
                value=torch_observation_minus_component_explained_by_other_features)

            # log_ratio = log_likelihoods_per_latent_equal_one[dish_idx] - log_likelihoods_per_latent_equal_zero[dish_idx]

            # reset mask for next step
            mask[dish_idx] = True

        # typically, we would compute the posterior as:
        # p(z=1|o, history) = p(z=1, o|history) / p(o|history)
        #                   = p(z=1, o |history) / (p(z=1, o |history) + p(z=0, o |history))
        # this is numerically unstable. Instead, we use the following likelihood ratio-based approach
        # p(z=1|o, history) / p(z=0|o, history) = p(o|z=1) p(z=1|history) / (p(o|z=0) p(z=0|history))
        # rearranging, we see that
        # p(z=1|o, history) / p(z=0|o, history) =
        #               exp(log p(o|z=1) + log p(z=1|history) - log (p(o|z=0) - log p(z=0|history)))
        # Noting that p(z=0|o, history) = 1 - p(z=1|o, history), if we define the RHS as A,
        # then p(z=1|o, history) = A / (1 + A)
        A_argument = log_likelihoods_per_latent_equal_one - log_likelihoods_per_latent_equal_zero
        A_argument += torch.log(dish_eating_posterior) - torch.log(1 - dish_eating_posterior)
        A = torch.exp(A_argument)
        dish_eating_posterior[:] = torch.divide(A, 1 + A)

        # if A is infinite, we want the result to be 1 since inf/(1+inf) = 1
        dish_eating_posterior[torch.isinf(A)] = 1.

    return dish_eating_posterior


def recursive_ibp_compute_approx_lower_bound(torch_observation: torch.Tensor,
                                             obs_idx: int,
                                             dish_eating_prior: torch.Tensor,
                                             variational_params: Dict[str, torch.Tensor],
                                             sigma_obs_squared: float):
    logging.debug('entering:recursive_ibp_compute_approx_lower_bound')
    # convert half covariances to covariances
    prior_A_cov = utils.torch_helpers.convert_half_cov_to_cov(
        half_cov=variational_params['A']['half_cov'][obs_idx - 1])
    posterior_A_cov = utils.torch_helpers.convert_half_cov_to_cov(
        half_cov=variational_params['A']['half_cov'][obs_idx])

    indicators_term = utils.torch_helpers.expected_log_bernoulli_under_bernoulli(
        p_prob=dish_eating_prior,
        q_prob=variational_params['Z']['prob'][obs_idx])
    gaussian_term = utils.torch_helpers.expected_log_gaussian_under_gaussian(
        p_mean=variational_params['A']['mean'][obs_idx - 1],
        p_cov=prior_A_cov,
        q_mean=variational_params['A']['mean'][obs_idx],
        q_cov=posterior_A_cov)
    likelihood_term = utils.torch_helpers.expected_log_gaussian_under_linear_gaussian(
        observation=torch_observation,
        q_A_mean=variational_params['A']['mean'][obs_idx],
        q_A_cov=posterior_A_cov,
        q_Z_mean=variational_params['Z']['prob'][obs_idx],
        sigma_obs_squared=sigma_obs_squared)
    bernoulli_entropy = utils.torch_helpers.entropy_bernoulli(
        probs=variational_params['Z']['prob'][obs_idx])
    gaussian_entropy = utils.torch_helpers.entropy_gaussian(
        mean=variational_params['A']['mean'][obs_idx],
        cov=posterior_A_cov)

    lower_bound = indicators_term + gaussian_term + likelihood_term + bernoulli_entropy + gaussian_entropy

    utils.torch_helpers.assert_no_nan_no_inf(lower_bound)
    logging.debug('exiting:recursive_ibp_compute_approx_lower_bound')
    return lower_bound


def recursive_ibp_optimize_bernoulli_params(torch_observation: torch.Tensor,
                                            obs_idx: int,
                                            vi_idx: int,
                                            dish_eating_prior: torch.Tensor,
                                            variational_params: Dict[str, dict],
                                            sigma_obs_squared: int = 1e-0,
                                            simultaneous_or_sequential: str = 'sequential') -> torch.Tensor:
    assert simultaneous_or_sequential in {'sequential', 'simultaneous'}

    A_cov = utils.torch_helpers.convert_half_cov_to_cov(
        variational_params['A']['half_cov'][obs_idx, :])

    num_features = A_cov.shape[0]

    if simultaneous_or_sequential == 'simultaneous':
        slices_indices = [slice(0, num_features, 1)]
    elif simultaneous_or_sequential == 'sequential':
        slices_indices = [slice(k_idx, k_idx + 1, 1) for k_idx in range(num_features)]
        # switch up the order every now and again
        if vi_idx % 2 == 1:
            slices_indices = list(reversed(slices_indices))
    else:
        raise ValueError(f'Impermissible value for simultaneous_or_sequential: {simultaneous_or_sequential}')

    bernoulli_probs = variational_params['Z']['prob'][obs_idx].detach().clone()
    # either do 1 iteration of all indices (simultaneous) or do K iterations of each index (sequential)
    for slice_idx in slices_indices:
        log_bernoulli_prior_term = torch.log(
            torch.divide(dish_eating_prior[slice_idx],
                         1. - dish_eating_prior[slice_idx]))

        # -2 mu_{nk}^T o_n
        term_one = -2. * torch.einsum(
            'bk,k->b',
            variational_params['A']['mean'][obs_idx, slice_idx],
            torch_observation)

        # Tr[\Sigma_{nk} + \mu_{nk} \mu_{nk}^T]
        term_two = torch.einsum(
            'bii->b',
            torch.add(A_cov[slice_idx],
                      torch.einsum('bi,bj->bij',
                                   variational_params['A']['mean'][obs_idx, slice_idx],
                                   variational_params['A']['mean'][obs_idx, slice_idx])))

        # \mu_{nk}^T (\sum_{k': k' \neq k} b_{nk'} \mu_{nk'})
        # = \mu_{nk}^T (\sum_{k'} b_{nk'} \mu_{nk'}) - b_{nk} \mu_{nk}^T \mu_{nk}
        term_three_all_pairs = torch.einsum(
            'bi, i->b',
            variational_params['A']['mean'][obs_idx, slice_idx],
            torch.einsum(
                'b, bi->i',
                bernoulli_probs,
                variational_params['A']['mean'][obs_idx, :]))
        term_three_self_pairs = torch.einsum(
            'b,bk,bk->b',
            bernoulli_probs[slice_idx],
            variational_params['A']['mean'][obs_idx, slice_idx],
            variational_params['A']['mean'][obs_idx, slice_idx])
        # TODO: I think this 2 belongs here
        term_three = 2. * term_three_all_pairs - term_three_self_pairs

        # num_features = dish_eating_prior.shape[0]
        # mu = variable_variational_params['A']['mean'][obs_idx, slice_idx]
        # b = variable_variational_params['Z']['prob'][obs_idx, slice_idx]
        # term_three_check = torch.stack([
        #     torch.inner(mu[k],
        #                 torch.sum(torch.stack([b[kprime] * mu[kprime]
        #                                        for kprime in range(num_features)
        #                                        if kprime != k]),
        #                           dim=0))
        #     for k in range(num_features)
        # ])
        # assert torch.allclose(term_three, term_three_check)

        term_to_exponentiate = log_bernoulli_prior_term - 0.5 * (term_one + term_two + term_three) / sigma_obs_squared
        bernoulli_probs[slice_idx] = 1. / (1. + torch.exp(-term_to_exponentiate))

    # check that Bernoulli probs are all valid
    utils.torch_helpers.assert_no_nan_no_inf(bernoulli_probs)
    assert torch.all(0. <= bernoulli_probs)
    assert torch.all(bernoulli_probs <= 1.)

    return bernoulli_probs


def recursive_ibp_optimize_gaussian_params(torch_observation: torch.Tensor,
                                           obs_idx: int,
                                           vi_idx: int,
                                           variable_variational_params: Dict[str, dict],
                                           sigma_obs_squared: int = 1e-0,
                                           simultaneous_or_sequential: str = 'sequential',
                                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    assert sigma_obs_squared > 0
    assert simultaneous_or_sequential in {'sequential', 'simultaneous'}

    prev_means = variable_variational_params['A']['mean'][obs_idx - 1, :]
    prev_covs = utils.torch_helpers.convert_half_cov_to_cov(
        variable_variational_params['A']['half_cov'][obs_idx - 1, :])
    prev_precisions = torch.linalg.inv(prev_covs)

    obs_dim = torch_observation.shape[0]
    num_features = prev_means.shape[0]

    # Step 1: Compute updated covariances
    # Take I_{D \times D} and repeat to add a batch dimension
    # Resulting object has shape (num_features, obs_dim, obs_dim)
    repeated_eyes = torch.eye(obs_dim).reshape(1, obs_dim, obs_dim).repeat(num_features, 1, 1)
    weighted_eyes = torch.multiply(
        variable_variational_params['Z']['prob'][obs_idx, :, None, None],  # shape (num features, 1, 1)
        repeated_eyes) / sigma_obs_squared
    gaussian_precisions = torch.add(prev_precisions, weighted_eyes)
    gaussian_covs = torch.linalg.inv(gaussian_precisions)

    # no update on pytorch matrix square root
    # https://github.com/pytorch/pytorch/issues/9983#issuecomment-907530049
    # https://github.com/pytorch/pytorch/issues/25481
    gaussian_half_covs = torch.stack([
        torch.from_numpy(scipy.linalg.sqrtm(gaussian_cov.detach().numpy()))
        for gaussian_cov in gaussian_covs])

    # Step 2: Use updated covariances to compute updated means
    # Sigma_{n-1,l}^{-1} \mu_{n-1, l}
    term_one = torch.einsum(
        'aij,aj->ai',
        prev_precisions,
        prev_means)
    # b_{nl} (o_n - \sum_{k: k\neq l} b_{nk} \mu_{nk})

    gaussian_means = variable_variational_params['A']['mean'][obs_idx, :].detach().clone()
    # prev_gaussian_means = gaussian_means.detach().clone()

    # The covariance updates only depends on the previous covariance and Z_n, so we can always
    # update them independently of one another. The mean updates are trickier since they
    # depend on one another. We have two choices: update simultaneously or sequentially.
    # Simultaneous appears to be correct, based on math. However, simultaneous updates can lead
    # to a pathology: the inferred means will oscillate and never converge. What happens is if 2
    # features summed are too big, then both shrink, and then their sum is too small, so they both
    # grow. This can repeat forever.
    if simultaneous_or_sequential == 'simultaneous':
        slices_indices = [slice(0, num_features, 1)]
    elif simultaneous_or_sequential == 'sequential':
        slices_indices = [slice(k_idx, k_idx + 1, 1) for k_idx in range(num_features)]
        # switch up the order every now and again
        if vi_idx % 2 == 1:
            slices_indices = list(reversed(slices_indices))
    else:
        raise ValueError(f'Impermissible value for simultaneous_or_sequential: {simultaneous_or_sequential}')

    indices_per_slice = int(num_features / len(slices_indices))
    # either do 1 iteration of all indices (simultaneous) or do K iterations of each index (sequential)
    for slice_idx in slices_indices:
        weighted_all_curr_means = torch.multiply(
            gaussian_means,
            variable_variational_params['Z']['prob'][obs_idx, :, None])
        assert weighted_all_curr_means.shape == (num_features, obs_dim)
        weighted_non_l_curr_means = torch.subtract(
            torch.sum(weighted_all_curr_means, dim=0)[None, :],
            weighted_all_curr_means[slice_idx])

        obs_minus_weighted_non_l_means = torch.subtract(
            torch_observation,
            weighted_non_l_curr_means)
        assert obs_minus_weighted_non_l_means.shape == (indices_per_slice, obs_dim)

        term_two_l = torch.multiply(
            obs_minus_weighted_non_l_means,
            variable_variational_params['Z']['prob'][obs_idx, slice_idx, None]) / sigma_obs_squared
        assert term_two_l.shape == (indices_per_slice, obs_dim)

        gaussian_means[slice_idx] = torch.einsum(
            'aij,aj->ai',
            gaussian_covs[slice_idx, :, :],
            torch.add(term_one[slice_idx], term_two_l))
        assert gaussian_means[slice_idx].shape == (indices_per_slice, obs_dim)

    # gaussian_update_norm = torch.linalg.norm(gaussian_means - prev_gaussian_means)
    utils.torch_helpers.assert_no_nan_no_inf(gaussian_means)
    utils.torch_helpers.assert_no_nan_no_inf(gaussian_half_covs)
    return gaussian_means, gaussian_half_covs


def run_inference_alg(inference_alg_str: str,
                      observations: np.ndarray,
                      model_str: str,
                      model_params: dict = None,
                      plot_dir: str = None):
    # create dict to store algorithm-specific arguments to inference alg function
    if model_params is None:
        model_params = {}

    if model_str == 'linear_gaussian':
        model_params['gaussian_likelihood_cov_scaling'] = 1.
        model_params['gaussian_prior_cov_scaling'] = 100.
    elif model_str == 'factor_analysis':
        raise NotImplementedError
    elif model_str == 'nonnegative_matrix_factorization':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # select inference alg and add kwargs as necessary
    if inference_alg_str == 'Doshi-Velez':
        # TODO: investigate what these are
        model_params['t0'] = 1
        model_params['kappa'] = 0.5
        num_coordinate_ascent_steps = 17
        logging.info(f'Number of coordinate ascent steps: '
                     f'{num_coordinate_ascent_steps}')
        inference_alg = DoshiVelezLinearGaussian(
            model_str=model_str,
            model_params=model_params,
            num_coordinate_ascent_steps=num_coordinate_ascent_steps)
    elif inference_alg_str == 'R-IBP':
        inference_alg = RecursiveIBPLinearGaussian(
            model_str=model_str,
            model_params=model_params,
            plot_dir=None,
            # plot_dir=plot_dir,
        )
    elif inference_alg_str == 'Widjaja':
        # TODO: investigate what these are
        model_params['t0'] = 1
        model_params['kappa'] = 0.5
        inference_alg = WidjajaLinearGaussian(
            model_str=model_str,
            model_params=model_params)
    # elif inference_alg_str == 'Online CRP':
    #     inference_alg = online_crp
    # elif inference_alg_str == 'SUSG':
    #     inference_alg = sequential_updating_and_greedy_search
    # elif inference_alg_str == 'VSUSG':
    #     inference_alg = variational_sequential_updating_and_greedy_search
    # elif inference_alg_str.startswith('DP-Means'):
    #     inference_alg = dp_means
    #     if inference_alg_str.endswith('(offline)'):
    #         inference_alg_kwargs['num_passes'] = 8  # same as Kulis and Jordan
    #     elif inference_alg_str.endswith('(online)'):
    #         inference_alg_kwargs['num_passes'] = 1
    #     else:
    #         raise ValueError('Invalid DP Means')
    elif inference_alg_str.startswith('HMC-Gibbs'):
        inference_alg = HMCGibbsLinearGaussian(
            model_str=model_str,
            model_params=model_params,
            num_samples=100000,
            num_warmup_samples=50000,
            num_thinning_samples=1000)

        # Suppose inference_alg_str is 'HMC-Gibbs (5000 Samples)'. We want to extract
        # the number of samples. To do this, we use the following
        # num_samples = int(inference_alg_str.split(' ')[1][1:])

    # elif inference_alg_str.startswith('SVI'):
    #     inference_alg = stochastic_variational_inference
    #     learning_rate = 5e-4
    #     # suppose the inference_alg_str is 'SVI (5k Steps)'
    #     substrs = inference_alg_str.split(' ')
    #     num_steps = 1000 * int(substrs[1][1:-1])
    #     inference_alg_kwargs['num_steps'] = num_steps
    #     # Note: these are the ground truth params
    #     if likelihood_model == 'dirichlet_multinomial':
    #         inference_alg_kwargs['model_params'] = dict(
    #             dirichlet_inference_params=10.)  # same as R-CRP
    #     elif likelihood_model == 'multivariate_normal':
    #         inference_alg_kwargs['model_params'] = dict(
    #             gaussian_mean_prior_cov_scaling=6.,
    #             gaussian_cov_scaling=0.3)
    #     else:
    #         raise ValueError
    # elif inference_alg_str.startswith('Variational Bayes'):
    #     inference_alg = variational_bayes
    #     # Suppose we have an algorithm string 'Variational Bayes (10 Init, 10 Iterations)',
    #     substrs = inference_alg_str.split(' ')
    #     num_initializations = int(substrs[2][1:])
    #     max_iters = int(substrs[4])
    #     inference_alg_kwargs['num_initializations'] = num_initializations
    #     inference_alg_kwargs['max_iter'] = max_iters
    else:
        raise ValueError(f'Unknown inference algorithm: {inference_alg_str}')

    # run inference algorithm
    inference_alg_results = inference_alg.fit(
        observations=observations)

    # Add inference alg object to results, for later generating predictions
    inference_alg_results['inference_alg'] = inference_alg

    return inference_alg_results
