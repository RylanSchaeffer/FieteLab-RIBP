import abc
import jax
import jax.random
import numpy as np
import numpyro
import torch
from typing import Dict, Tuple, Union

import utils.numpy_helpers
import utils.numpyro_models
import utils.torch_helpers

torch.set_default_tensor_type('torch.FloatTensor')


def compute_max_num_features(alpha: float,
                             beta: float,
                             num_obs: int,
                             prefactor: int = 10):
    # Note: the expected number of latents grows logarithmically as a*b*log(1 + N/sticks)
    # The 10 is a hopefully conservative heuristic to preallocate.
    return prefactor * int(alpha * beta * np.log(1 + num_obs / beta))


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

        # we use half covariance in case we want to numerically optimize
        w_half_covs = torch.stack([

        ])

        A_half_covs = torch.stack([
            np.sqrt(self.gen_model_params['feature_prior_params']['gaussian_mean_prior_cov_scaling']) * torch.eye(
                obs_dim).float()
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
            w=dict(
                mean=torch.full(
                    size=(num_obs + 1, max_num_features),
                    fill_value=0.,
                    dtype=torch.float32,
                ),
                half_cov=w_half_covs,
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
            num_dishes_poisson_rate_priors[obs_idx, :] += self.gen_model_params['IBP']['alpha'] \
                                                          * self.gen_model_params['IBP']['beta'] \
                                                          / (self.gen_model_params['IBP']['beta'] + obs_idx - 1)
            # Recursion: 1st term
            dish_eating_prior = torch.clone(
                dish_eating_posteriors_running_sum[obs_idx - 1, :]) / (
                                            self.gen_model_params['IBP']['beta'] + obs_idx - 1)
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
                            sigma_obs_squared=self.gen_model_params['likelihood_params']['sigma_x'] ** 2)
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
                            sigma_obs_squared=self.gen_model_params['likelihood_params']['sigma_x'] ** 2)
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
            gen_model_params=self.gen_model_params,
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

