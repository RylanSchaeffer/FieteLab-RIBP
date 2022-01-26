import abc
import jax
import jax.random
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import os
import scipy
from scipy.stats import poisson
import torch
from typing import Dict, Tuple, Union

import utils.numpy_helpers
import utils.numpyro_models
import utils.torch_helpers

torch.set_default_tensor_type('torch.FloatTensor')


def compute_max_num_features(alpha: float,
                             beta: float,
                             num_obs: int,
                             prefactor: int = 2):
    # Note: the expected number of latents grows logarithmically as a*b*log(1 + N/sticks)
    # The 2 is a hopefully conservative heuristic to preallocate.
    # Note: Add 1 to ensure at least one feature exists.
    return prefactor * int(1 + alpha * beta * np.log(1 + num_obs / beta))


class FactorAnalysisModel(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.model_str = None
        self.gen_model_params = None
        self.plot_dir = None
        self.fit_results = None

    @abc.abstractmethod
    def fit(self,
            observations: np.ndarray):
        pass

    @abc.abstractmethod
    def sample_variables_for_predictive_posterior(self,
                                                  num_samples: int):
        pass

    @abc.abstractmethod
    def features_after_last_obs(self) -> np.ndarray:
        """
        Returns array of shape (num features, feature dimension)
        """
        pass


class HMCGibbsFactorAnalysis(FactorAnalysisModel):

    def __init__(self,
                 model_str: str,
                 model_params: Dict[str, float],
                 num_samples: int,
                 num_warmup_samples: int,
                 num_thinning_samples: int,
                 max_num_features: int = None):

        assert model_str in {'linear_gaussian', 'factor_analysis',
                             'nonnegative_matrix_factorization'}
        assert 'alpha' in model_params['IBP']
        assert 'beta' in model_params['IBP']
        assert model_params['IBP']['alpha'] > 0
        assert model_params['IBP']['beta'] > 0

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
                alpha=self.model_params['IBP']['alpha'],
                beta=self.model_params['IBP']['beta'],
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

    def sample_variables_for_predictive_posterior(self,
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


class RecursiveIBPFactorAnalysis(FactorAnalysisModel):

    def __init__(self,
                 model_str: str,
                 gen_model_params: Dict[str, Dict[str, float]],
                 ossify_features: bool = True,
                 numerically_optimize: bool = False,
                 learning_rate: float = 1e0,
                 coord_ascent_update_type: str = 'sequential',
                 num_coord_ascent_steps_per_obs: int = 3,
                 plot_dir: str = None):

        # Check validity of input params
        assert model_str == 'factor_analysis'
        self.gen_model_params = gen_model_params
        self.ibp_params = gen_model_params['IBP']
        assert self.ibp_params['alpha'] > 0
        assert self.ibp_params['beta'] > 0
        self.feature_prior_params = gen_model_params['feature_prior_params']
        self.scale_prior_params = gen_model_params['scale_prior_params']
        self.likelihood_params = gen_model_params['likelihood_params']
        assert coord_ascent_update_type in {'simultaneous', 'sequential'}
        if numerically_optimize:
            assert learning_rate is not None
            assert learning_rate > 0.

        if numerically_optimize is False:
            learning_rate = np.nan
        else:
            assert isinstance(learning_rate, float)
            assert learning_rate > 0.

        self.model_str = model_str
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
            alpha=self.gen_model_params['IBP']['alpha'],
            beta=self.gen_model_params['IBP']['beta'],
            num_obs=num_obs)

        # The recursion does not require recording the full history of priors/posteriors
        # but we record the full history for subsequent analysis.
        dish_eating_priors = torch.zeros(
            (num_obs + 1, max_num_features),  # Add 1 to the number of observations to use 1-based indexing
            dtype=torch.float32,
            requires_grad=self.numerically_optimize)

        # dish_eating_posteriors are contained in the variational
        # parameters ['Z']['probs']

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

        Sigma_w_inv = torch.eye(max_num_features) \
                      / self.gen_model_params['scale_prior_params']['scale_prior_cov_scaling']

        # we use half covariance in case we want to numerically optimize
        w_prefactor = np.sqrt(self.gen_model_params['scale_prior_params']['scale_prior_cov_scaling'])
        w_half_cov = (w_prefactor * torch.eye(max_num_features).float()[None, :, :]).repeat(
            num_obs + 1, 1, 1,)

        # we use half covariance because we want to numerically optimize
        A_prefactor = np.sqrt(self.gen_model_params['feature_prior_params']['feature_prior_cov_scaling'])
        A_half_covs = (A_prefactor * torch.eye(obs_dim).float()[None, None, :, :]).repeat(
            1, max_num_features, 1, 1,)

        # dict mapping variables to variational params
        self.variational_params = dict(
            Z=dict(  # variational params for binary indicators
                prob=torch.full(
                    size=(num_obs + 1, max_num_features),
                    fill_value=np.nan,
                    dtype=torch.float32),
            ),
            w=dict(
                mean=torch.from_numpy(np.random.normal(
                    loc=0,
                    scale=1.5,
                    size=(num_obs + 1, max_num_features)).astype(dtype=np.float32)),
                half_cov=w_half_cov,
            ),
            A=dict(  # variational params for Gaussian features
                mean=torch.full(
                    size=(1, max_num_features, obs_dim),
                    fill_value=0.,
                    dtype=torch.float32),
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
            num_dishes_poisson_rate_priors[obs_idx, :] += self.gen_model_params['IBP']['alpha'] \
                                                          * self.gen_model_params['IBP']['beta'] \
                                                          / (self.gen_model_params['IBP']['beta'] + obs_idx - 1)
            # Recursion: 1st term
            dish_eating_prior = torch.clone(
                dish_eating_posteriors_running_sum[obs_idx - 1, :]) / (
                                        self.gen_model_params['IBP']['beta'] + obs_idx - 1)
            # Recursion: 2nd term; don't subtract 1 from latent indices b/c zero based indexing
            dish_eating_prior += poisson.cdf(
                k=latent_indices,
                mu=num_dishes_poisson_rate_posteriors[obs_idx - 1, :])
            # Recursion: 3rd term; don't subtract 1 from latent indices b/c zero based indexing
            dish_eating_prior -= poisson.cdf(
                k=latent_indices,
                mu=num_dishes_poisson_rate_priors[obs_idx, :])

            # record latent prior
            dish_eating_priors[obs_idx, :] = dish_eating_prior.clone()

            # Initialize dish eating posterior to dish eating prior, before
            # beginning inference.
            self.variational_params['Z']['prob'].data[obs_idx, :] = dish_eating_prior.clone()

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
                    # # maximize approximate lower bound with respect to A's params
                    # approx_lower_bound = recursive_ibp_compute_approx_lower_bound(
                    #     torch_observation=torch_observation,
                    #     obs_idx=obs_idx,
                    #     dish_eating_prior=dish_eating_prior,
                    #     variational_params=self.variational_params)
                    # approx_lower_bounds.append(approx_lower_bound.item())
                    # approx_lower_bound.backward()
                    #
                    # # scale learning rate by posterior(A_k) / sum_n prev_posteriors(A_k)
                    # # scale by 1/num_vi_steps so that after num_vi_steps, we've moved O(1)
                    # scaled_learning_rate = self.learning_rate * torch.divide(
                    #     dish_eating_posteriors[obs_idx, :],
                    #     dish_eating_posteriors[obs_idx, :] + dish_eating_posteriors_running_sum[obs_idx - 1, :])
                    # scaled_learning_rate /= self.num_coord_ascent_steps_per_obs
                    # scaled_learning_rate[torch.isnan(scaled_learning_rate)] = 0.
                    # scaled_learning_rate[torch.isinf(scaled_learning_rate)] = 0.
                    #
                    # # make sure no gradient when applying gradient updates
                    # with torch.no_grad():
                    #     for var_name, var_dict in self.variational_params.items():
                    #         for param_name, param_tensor in var_dict.items():
                    #             # the scaled learning rate has shape (num latents,) aka (num obs,)
                    #             # we need to create extra dimensions of size 1 for broadcasting to work
                    #             # because param_tensor can have variable number of dimensions e.g. (num obs, obs dim) for mean
                    #             # or (num obs, obs dim, obs dim) for covariance, we need to dynamically
                    #             # add the correct number of dimensions
                    #             # Also, exclude dimension 0 because that's for the observation index
                    #             reshaped_scaled_learning_rate = scaled_learning_rate.view(
                    #                 [param_tensor.shape[1]] + [1 for _ in range(len(param_tensor.shape[2:]))])
                    #             if param_tensor.grad is None:
                    #                 continue
                    #             else:
                    #                 scaled_param_tensor_grad = torch.multiply(
                    #                     reshaped_scaled_learning_rate,
                    #                     param_tensor.grad[obs_idx, :])
                    #                 param_tensor.data[obs_idx, :] += scaled_param_tensor_grad
                    #                 utils.torch_helpers.assert_no_nan_no_inf(param_tensor.data[:obs_idx + 1])
                    #
                    #                 # zero gradient manually
                    #                 param_tensor.grad = None

                    raise NotImplementedError

                elif not self.numerically_optimize:
                    with torch.no_grad():
                        print(f'Obs Idx: {obs_idx}, VI idx: {vi_idx}')

                        A_means, A_half_covs = recursive_ibp_optimize_feature_params(
                            torch_observation=torch_observation,
                            obs_idx=obs_idx,
                            vi_idx=vi_idx,
                            variable_variational_params=self.variational_params,
                            simultaneous_or_sequential=self.coord_ascent_update_type,
                            sigma_obs_squared=self.gen_model_params['likelihood_params']['sigma_x'] ** 2)

                        if self.ossify_features:
                            # TODO: refactor into own function
                            normalizing_const = torch.add(
                                self.variational_params['Z']['prob'].data[obs_idx, :],
                                dish_eating_posteriors_running_sum[obs_idx - 1, :])
                            history_weighted_A_means = torch.add(
                                torch.multiply(self.variational_params['Z']['prob'].data[obs_idx, :, None],
                                               A_means),
                                torch.multiply(dish_eating_posteriors_running_sum[obs_idx - 1, :, None],
                                               self.variational_params['A']['mean'].data[0, :]))
                            history_weighted_A_half_covs = torch.add(
                                torch.multiply(self.variational_params['Z']['prob'].data[obs_idx, :, None, None],
                                               A_half_covs),
                                torch.multiply(dish_eating_posteriors_running_sum[obs_idx - 1, :, None, None],
                                               self.variational_params['A']['half_cov'].data[0, :]))
                            A_means = torch.divide(history_weighted_A_means, normalizing_const[:, None])
                            A_half_covs = torch.divide(
                                history_weighted_A_half_covs,
                                normalizing_const[:, None, None])
                            # if cumulative probability mass is 0, we compute 0/0 and get NaN. Need to mask
                            A_means[normalizing_const == 0.] = \
                                self.variational_params['A']['mean'][0][normalizing_const == 0.]
                            A_half_covs[normalizing_const == 0.] = \
                                self.variational_params['A']['half_cov'][0][normalizing_const == 0.]

                            utils.torch_helpers.assert_no_nan_no_inf_is_real(A_means)
                            utils.torch_helpers.assert_no_nan_no_inf_is_real(A_half_covs)

                        self.variational_params['A']['mean'].data[0, :] = A_means
                        self.variational_params['A']['half_cov'].data[0, :] = A_half_covs

                        w_mean, w_half_cov = recursive_ibp_optimize_scale_params(
                            torch_observation=torch_observation,
                            obs_idx=obs_idx,
                            vi_idx=vi_idx,
                            variable_variational_params=self.variational_params,
                            simultaneous_or_sequential=self.coord_ascent_update_type,
                            Sigma_w_inv=Sigma_w_inv,
                            sigma_obs_squared=self.gen_model_params['likelihood_params']['sigma_x'] ** 2)

                        # TODO: check why this produces
                        # UserWarning: Casting complex values to real discards the imaginary part
                        self.variational_params['w']['mean'].data[obs_idx, :] = w_mean
                        self.variational_params['w']['half_cov'].data[obs_idx, :, :] = w_half_cov

                        # maximize approximate lower bound with respect to Z's params
                        Z_probs = recursive_ibp_optimize_bernoulli_params(
                            torch_observation=torch_observation,
                            obs_idx=obs_idx,
                            vi_idx=vi_idx,
                            dish_eating_prior=dish_eating_prior,
                            variable_variational_params=self.variational_params,
                            simultaneous_or_sequential=self.coord_ascent_update_type,
                            sigma_obs_squared=self.gen_model_params['likelihood_params']['sigma_x'] ** 2)

                        self.variational_params['Z']['prob'].data[obs_idx, :] = Z_probs

                else:
                    raise ValueError(f'Impermissible value of numerically_optimize: {self.numerically_optimize}')

                # if self.plot_dir is not None:
                #     fig.suptitle(f'Obs: {obs_idx}, {self.coord_ascent_update_type}')
                #     axes[vi_idx, 0].set_ylabel(f'VI Step: {vi_idx + 1}')
                #     axes[vi_idx, 0].set_title('Individual Features')
                #     axes[vi_idx, 0].scatter(observations[:obs_idx, 0],
                #                             observations[:obs_idx, 1],
                #                             # s=3,
                #                             color='k',
                #                             label='Observations')
                #     for feature_idx in range(10):
                #         axes[vi_idx, 0].plot(
                #             [0, self.variational_params['A']['mean'][0, feature_idx, 0].item()],
                #             [0, self.variational_params['A']['mean'][0, feature_idx, 1].item()],
                #             label=f'{feature_idx}')
                #     # axes[0].legend()
                #
                #     axes[vi_idx, 1].set_title('Inferred Indicator Probabilities')
                #     # axes[vi_idx, 1].set_xlabel('Indicator Index')
                #     axes[vi_idx, 1].scatter(
                #         1 + np.arange(10),
                #         dish_eating_priors[obs_idx, :10].detach().numpy(),
                #         label='Prior')
                #     axes[vi_idx, 1].scatter(
                #         1 + np.arange(10),
                #         dish_eating_posteriors[obs_idx, :10].detach().numpy(),
                #         label='Posterior')
                #     axes[vi_idx, 1].legend()
                #
                #     weighted_features = np.multiply(
                #         self.variational_params['A']['mean'][0, :, :].detach().numpy(),
                #         dish_eating_posteriors[obs_idx].unsqueeze(1).detach().numpy(),
                #     )
                #     axes[vi_idx, 2].set_title('Weighted Features')
                #     axes[vi_idx, 2].scatter(observations[:obs_idx, 0],
                #                             observations[:obs_idx, 1],
                #                             # s=3,
                #                             color='k',
                #                             label='Observations')
                #     for feature_idx in range(10):
                #         axes[vi_idx, 2].plot([0, weighted_features[feature_idx, 0]],
                #                              [0, weighted_features[feature_idx, 1]],
                #                              label=f'{feature_idx}',
                #                              zorder=feature_idx + 1,
                #                              # alpha=dish_eating_posteriors[obs_idx, feature_idx].item(),
                #                              )
                #
                #     cumulative_weighted_features = np.cumsum(weighted_features, axis=0)
                #     axes[vi_idx, 3].set_title('Cumulative Weighted Features')
                #     axes[vi_idx, 3].scatter(observations[:obs_idx, 0],
                #                             observations[:obs_idx, 1],
                #                             # s=3,
                #                             color='k',
                #                             label='Observations')
                #     for feature_idx in range(10):
                #         axes[vi_idx, 3].plot(
                #             [0 if feature_idx == 0 else cumulative_weighted_features[feature_idx - 1, 0],
                #              cumulative_weighted_features[feature_idx, 0]],
                #             [0 if feature_idx == 0 else cumulative_weighted_features[feature_idx - 1, 1],
                #              cumulative_weighted_features[feature_idx, 1]],
                #             label=f'{feature_idx}',
                #             alpha=dish_eating_posteriors[obs_idx, feature_idx].item())

                with torch.no_grad():

                    # update running sum of posteriors
                    dish_eating_posteriors_running_sum[obs_idx, :] = torch.add(
                        dish_eating_posteriors_running_sum[obs_idx - 1, :],
                        self.variational_params['Z']['prob'][obs_idx, :])

                    # update how many dishes have been sampled
                    non_eaten_dishes_posteriors_running_prod[obs_idx, :] = np.multiply(
                        non_eaten_dishes_posteriors_running_prod[obs_idx - 1, :],
                        1. - self.variational_params['Z']['prob'][obs_idx, :],
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
                        self.variational_params['Z']['prob'][obs_idx, :])

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
            var_name: {param_name: param_tensor.detach().numpy()
                       for param_name, param_tensor in var_params.items()}
            for var_name, var_params in self.variational_params.items()
        }

        # Chop off 0th observation index
        numpy_variable_params['Z']['prob'] = numpy_variable_params['Z']['prob'][1:]
        numpy_variable_params['w']['mean'] = numpy_variable_params['w']['mean'][1:]
        numpy_variable_params['w']['half_cov'] = numpy_variable_params['w']['half_cov'][1:]

        dish_eating_posteriors = numpy_variable_params['Z']['prob']

        self.fit_results = dict(
            dish_eating_priors=dish_eating_priors.numpy()[1:],
            dish_eating_posteriors=dish_eating_posteriors,  # already chopped
            dish_eating_posteriors_running_sum=dish_eating_posteriors_running_sum.numpy()[1:],
            non_eaten_dishes_posteriors_running_prod=non_eaten_dishes_posteriors_running_prod.numpy()[1:],
            num_dishes_poisson_rate_priors=num_dishes_poisson_rate_priors.numpy()[1:],
            num_dishes_poisson_rate_posteriors=num_dishes_poisson_rate_posteriors.numpy()[1:],
            variational_params=numpy_variable_params,
            gen_model_params=self.gen_model_params,
        )

        return self.fit_results

    def sample_variables_for_predictive_posterior(self,
                                                  num_samples: int) -> Dict[str, np.ndarray]:

        # obs index is customer index (1-based)
        # add one because we are predicting the next customer
        obs_idx = self.fit_results['dish_eating_posteriors'].shape[0] + 1
        max_num_features = self.fit_results['variational_params']['Z']['prob'].shape[1]
        latent_indices = np.arange(max_num_features)

        # Construct prior over next time step
        num_dishes_poisson_rate_prior = self.fit_results['num_dishes_poisson_rate_posteriors'][obs_idx - 2, :]
        num_dishes_poisson_rate_prior += self.gen_model_params['IBP']['alpha'] * self.gen_model_params['IBP']['beta'] \
                                         / (self.gen_model_params['IBP']['beta'] + obs_idx - 1)

        dish_eating_prior = self.fit_results['dish_eating_posteriors_running_sum'][obs_idx - 2, :] \
                            / (self.gen_model_params['IBP']['beta'] + obs_idx - 1)
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

        feature_covs = utils.numpy_helpers.convert_half_cov_to_cov(
            half_cov=var_params['A']['half_cov'][-1, :, :, :])
        features = np.stack([np.random.multivariate_normal(mean=var_params['A']['mean'][-1, k, :],
                                                           cov=feature_covs[k],
                                                           size=num_samples)
                             for k in range(max_num_features)])

        # Use the prior for w
        # Previous w_n don't matter
        w = np.random.multivariate_normal(
            mean=np.zeros(max_num_features),
            cov=self.gen_model_params['scale_prior_params']['scale_prior_cov_scaling'] * np.eye(max_num_features),
            size=num_samples,)

        # shape = (num samples, max num features, obs dim)
        features = features.transpose(1, 0, 2)

        # shape = (num samples, max num features)
        w = w.transpose(1, 0)

        sampled_params = dict(
            indicators_probs=indicators_probs,
            features=features,
            w=w)

        return sampled_params

    def features_after_last_obs(self) -> np.ndarray:
        return self.fit_results['variational_params']['A']['mean'][-1, :, :]


def recursive_ibp_optimize_bernoulli_params(torch_observation: torch.Tensor,
                                            obs_idx: int,
                                            vi_idx: int,
                                            dish_eating_prior: torch.Tensor,
                                            variable_variational_params: Dict[str, dict],
                                            sigma_obs_squared: int = 1e-0,
                                            simultaneous_or_sequential: str = 'sequential',
                                            ) -> torch.Tensor:
    assert simultaneous_or_sequential in {'sequential', 'simultaneous'}

    # Add, then remove, batch dimension
    # shape (1, max num features, max num features)
    w_cov = utils.torch_helpers.convert_half_cov_to_cov(
        variable_variational_params['w']['half_cov'][obs_idx:obs_idx + 1, :, :])[0, :, :]

    A_cov = utils.torch_helpers.convert_half_cov_to_cov(
        variable_variational_params['A']['half_cov'][0, :])

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

    bernoulli_probs = variable_variational_params['Z']['prob'][obs_idx].detach().clone()
    # either do 1 iteration of all indices (simultaneous) or do K iterations of each index (sequential)
    for slice_idx in slices_indices:
        # q(z_{nl}|o_{<n}) / (1 - q(z_{nl}|o_{<n}) )
        log_bernoulli_prior_term = torch.log(
            torch.divide(dish_eating_prior[slice_idx],
                         1. - dish_eating_prior[slice_idx]))

        # -2 o_n^T mu_{nl} phi_{nl}
        # shape (slice length,)
        term_one = -2. * variable_variational_params['w']['mean'][obs_idx, slice_idx] * torch.einsum(
            'kd,d->',
            variable_variational_params['A']['mean'][0, slice_idx],  # shape (1, obs dim)
            torch_observation,  # not sure if transpose needed
        )

        # [phi_nl^2 + Phi_nll] Tr[\Sigma_{nl} + \mu_{nl} \mu_{nl}^T]
        term_two_prefactor = torch.add(
            torch.diagonal(w_cov[slice_idx, slice_idx]),  # shape (slice length,)
            variable_variational_params['w']['mean'][obs_idx, slice_idx] ** 2.,
        )
        # shape (1,)
        term_two = term_two_prefactor * torch.einsum(
            'kii->k',
            torch.add(A_cov[slice_idx],
                      torch.einsum('ki,kj->kij',
                                   variable_variational_params['A']['mean'][0, slice_idx],
                                   variable_variational_params['A']['mean'][0, slice_idx])))

        # (\mu_{nl}^T \sum_{k: k \neq l} b_{nk} \phi_{nk} \mu_{nk} )
        # = (\mu_{nl}^T \sum_{k} b_{nk} \phi_{nk} \mu_{nk}) - (b_{nl} \phi_{nl} \mu_{nl}^T \mu_{nl})
        # shape (slice length, 1)
        term_three_all_pairs = torch.einsum(
            'bi,i->b',
            variable_variational_params['A']['mean'][0, slice_idx],  # shape (slice, obs dim)
            torch.einsum(
                'b,bi->i',
                torch.mul(bernoulli_probs, variable_variational_params['w']['mean'][obs_idx]),
                # shape (max num features)
                variable_variational_params['A']['mean'][0, :]))
        # shape (slice length, 1)
        term_three_self_pairs = torch.einsum(
            'b,bk,bk->b',
            torch.mul(bernoulli_probs[slice_idx],
                      variable_variational_params['w']['mean'][obs_idx][slice_idx]),
            variable_variational_params['A']['mean'][0, slice_idx],
            variable_variational_params['A']['mean'][0, slice_idx])

        term_three = 2. * variable_variational_params['w']['mean'][obs_idx, slice_idx] * (
                term_three_all_pairs - term_three_self_pairs)

        # num_features = dish_eating_prior.shape[0]
        # mu = variational_params['A']['mean'][obs_idx, :]
        # b = variational_params['Z']['prob'][obs_idx, :]
        # TODO: Change 0 index to slice index
        # term_three_check = 2. * torch.inner(
        #     mu[0],
        #     torch.sum(torch.stack([b[kprime] * mu[kprime]
        #                            for kprime in range(num_features)
        #                            if kprime != 0]),
        #               dim=0))
        # assert torch.allclose(term_three, term_three_check)

        term_to_exponentiate = log_bernoulli_prior_term - 0.5 * (
                term_one + term_two + term_three) / sigma_obs_squared
        bernoulli_probs[slice_idx] = 1. / (1. + torch.exp(-term_to_exponentiate))

    # check that Bernoulli probs are all valid
    utils.torch_helpers.assert_no_nan_no_inf_is_real(bernoulli_probs)
    assert torch.all(0. <= bernoulli_probs)
    assert torch.all(bernoulli_probs <= 1.)

    # Shape: (Max num features,)
    return bernoulli_probs


def recursive_ibp_optimize_feature_params(torch_observation: torch.Tensor,
                                          obs_idx: int,
                                          vi_idx: int,
                                          variable_variational_params: Dict[str, dict],
                                          sigma_obs_squared: int = 1e-0,
                                          simultaneous_or_sequential: str = 'sequential',
                                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    assert sigma_obs_squared > 0
    assert simultaneous_or_sequential in {'sequential', 'simultaneous'}

    prev_means = variable_variational_params['A']['mean'][0, :].clone()
    prev_covs = utils.torch_helpers.convert_half_cov_to_cov(
        variable_variational_params['A']['half_cov'][0, :])
    prev_precisions = torch.linalg.inv(prev_covs)

    # shape: (max num features, max number features)
    w_covs = utils.torch_helpers.convert_half_cov_to_cov(
        variable_variational_params['w']['half_cov'][obs_idx:obs_idx + 1, :, :])[0, :, :]

    obs_dim = torch_observation.shape[0]
    num_features = prev_means.shape[0]

    # Step 1: Compute updated covariances
    # Take I_{D \times D} and repeat to add a batch dimension
    # Resulting object has shape (num_features, obs_dim, obs_dim)
    repeated_eyes = torch.eye(obs_dim).reshape(1, obs_dim, obs_dim).repeat(num_features, 1, 1)

    # Matrix form of \Phi_{nll} + \phi_{nl}^2
    # For the covariance diagonal, I remove batch dimension then add it back in
    # I can't find a batch version of torch.diagonal
    # shape = (max num features, )
    phi_term = torch.add(
        torch.diagonal(w_covs),
        torch.multiply(variable_variational_params['w']['mean'][obs_idx, :],
                       variable_variational_params['w']['mean'][obs_idx, :]))

    weighted_eyes = torch.multiply(
        torch.multiply(variable_variational_params['Z']['prob'][obs_idx, :],
                       phi_term)[:, None, None, ],  # shape (1, num features, 1, 1)
        repeated_eyes) / sigma_obs_squared  # unsure about dimensions
    gaussian_precisions = torch.add(prev_precisions, weighted_eyes)
    gaussian_covs = torch.linalg.inv(gaussian_precisions)

    # no update on pytorch matrix square root
    # https://github.com/pytorch/pytorch/issues/9983#issuecomment-907530049
    # https://github.com/pytorch/pytorch/issues/25481
    feature_half_covs = torch.stack([
        torch.from_numpy(scipy.linalg.sqrtm(gaussian_cov.detach().numpy()))
        for gaussian_cov in gaussian_covs])

    # Step 2: Use updated covariances to compute updated means
    # Sigma_{n-1,l}^{-1} \mu_{n-1, l}
    # shape (max num features, obs dim)
    term_one = torch.einsum(
        'aij,aj->ai',
        prev_precisions,
        prev_means)
    # b_{nl} (o_n - \sum_{k: k\neq l} b_{nk} \mu_{nk})

    feature_means = variable_variational_params['A']['mean'][0, :].detach().clone()
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
            feature_means,
            torch.multiply(variable_variational_params['Z']['prob'][obs_idx, :, None],
                           variable_variational_params['w']['mean'][obs_idx, :, None]),  # shape (max num features, 1)
        )
        assert weighted_all_curr_means.shape == (num_features, obs_dim)
        weighted_non_l_curr_means = torch.subtract(
            torch.sum(weighted_all_curr_means, dim=0)[None, :],
            weighted_all_curr_means[slice_idx])

        obs_minus_weighted_non_l_means = torch.subtract(
            torch_observation,
            weighted_non_l_curr_means)
        assert obs_minus_weighted_non_l_means.shape == (indices_per_slice, obs_dim)

        term_two_l = torch.multiply(
            obs_minus_weighted_non_l_means,  # shape (obs dim, 1)
            torch.multiply(variable_variational_params['Z']['prob'][obs_idx, slice_idx, None],
                           variable_variational_params['w']['mean'][obs_idx, slice_idx, None])) / sigma_obs_squared
        assert term_two_l.shape == (indices_per_slice, obs_dim)

        feature_means[slice_idx] = torch.einsum(
            'aij,aj->ai',
            gaussian_covs[slice_idx, :, :],
            torch.add(term_one[slice_idx], term_two_l))
        assert feature_means[slice_idx].shape == (indices_per_slice, obs_dim)

    # gaussian_update_norm = torch.linalg.norm(gaussian_means - prev_gaussian_means)
    utils.torch_helpers.assert_no_nan_no_inf_is_real(feature_means)
    utils.torch_helpers.assert_no_nan_no_inf_is_real(feature_half_covs)
    return feature_means, feature_half_covs


def recursive_ibp_optimize_scale_params(torch_observation: torch.Tensor,
                                        obs_idx: int,
                                        vi_idx: int,
                                        variable_variational_params: Dict[str, dict],
                                        Sigma_w_inv: torch.Tensor,
                                        sigma_obs_squared: int = 1e-0,
                                        simultaneous_or_sequential: str = 'sequential',
                                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    assert sigma_obs_squared > 0
    assert simultaneous_or_sequential in {'sequential', 'simultaneous'}

    obs_dim = torch_observation.shape[0]
    num_features = variable_variational_params['Z']['prob'].shape[1]

    # shape (num features, obs dim)
    A_means = variable_variational_params['A']['mean'][0, :]
    # shape (num features, obs dim, obs dim)
    A_covs = utils.torch_helpers.convert_half_cov_to_cov(
        variable_variational_params['A']['half_cov'][0, :, :])  # DxD
    bernoulli_probs = variable_variational_params['Z']['prob'][obs_idx].detach().clone()

    # Step 1: Compute updated covariances
    M = torch.zeros(size=(num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            M[i, j] = torch.inner(A_means[i], A_means[j])
            if i == j:
                M[i, j] += torch.trace(A_covs[i])

    S = torch.zeros(size=(num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            S[i, j] = M[i, j] * bernoulli_probs[i]
            if i != j:
                S[i, j] *= bernoulli_probs[j]

    # shape (num features, num features)
    scale_precisions = torch.add(
        Sigma_w_inv,
        S / sigma_obs_squared)

    # shape (max features, max features)
    scale_cov = torch.linalg.inv(scale_precisions)
    scale_half_cov = torch.from_numpy(
        scipy.linalg.sqrtm(scale_cov.detach().numpy()))

    # shape (max features, )
    scale_mean = torch.einsum(
        'ab,bc,cd,d->a',
        scale_cov,
        torch.diag(bernoulli_probs),
        A_means,  # shape (max features, obs dim)
        torch_observation,  # shape (obs dim,)
    ) / sigma_obs_squared

    # TODO: for some reason, scale_half_cov ends up with very small imaginary values
    # e.g. -2.8021e-17+1.8830e-21j,
    # Let's just remove this for now, figure out why later.
    if torch.is_complex(scale_half_cov):
        scale_half_cov = torch.real(scale_half_cov)

    utils.torch_helpers.assert_no_nan_no_inf_is_real(scale_mean)
    utils.torch_helpers.assert_no_nan_no_inf_is_real(scale_half_cov)

    # shape (num features) and (num features, num features)
    return scale_mean, scale_half_cov
