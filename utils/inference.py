import cvxpy as cp
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.linalg
from scipy.stats import poisson
from timeit import default_timer as timer
import torch
import tracemalloc
from typing import Dict, Tuple

import utils.torch_helpers

# torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_default_tensor_type('torch.FloatTensor')

inference_alg_strs = [
    'R-IBP',
]


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


def recursive_ibp(observations,
                  inference_params: dict,
                  likelihood_model: str,
                  variable_variational_params: dict = None,
                  num_vi_steps_per_obs: int = 3,
                  plot_dir: str = None,
                  simultaneous_or_sequential: str = 'sequential',
                  ossify_features: bool = True,
                  numerically_optimize: bool = False,
                  learning_rate: float = 1e0,
                  ):
    """
    Perform inference with given likelihood


    """

    assert simultaneous_or_sequential in {'simultaneous', 'sequential'}
    for param, param_value in inference_params.items():
        assert param_value > 0
    alpha = inference_params['alpha']
    beta = inference_params['beta']
    assert likelihood_model in {'multivariate_normal', 'dirichlet_multinomial',
                                'bernoulli', 'continuous_bernoulli', 'linear_gaussian'}
    if numerically_optimize:
        assert learning_rate is not None
        assert learning_rate > 0.

    num_obs, obs_dim = observations.shape

    # Note: the expected number of latents grows logarithmically as a*b*log(b + T - 1)
    # The 10 is a hopefully conservative heuristic to preallocate.
    max_num_features = 10 * int(inference_params['alpha'] * inference_params['beta'] * \
                                np.log(inference_params['beta'] + num_obs - 1))

    # The recursion does not require recording the full history of priors/posteriors
    # but we record the full history for subsequent analysis.
    dish_eating_priors = torch.zeros(
        (num_obs + 1, max_num_features),  # Add 1 to the number of observations to use 1-based indexing
        dtype=torch.float32,
        requires_grad=False)

    dish_eating_posteriors = torch.zeros(
        (num_obs + 1, max_num_features),
        dtype=torch.float32,
        requires_grad=False)

    dish_eating_posteriors_running_sum = torch.zeros(
        (num_obs + 1, max_num_features),
        dtype=torch.float32,
        requires_grad=False)

    non_eaten_dishes_posteriors_running_prod = torch.ones(
        (num_obs + 1, max_num_features),
        dtype=torch.float32,
        requires_grad=False)

    num_dishes_poisson_rate_priors = torch.zeros(
        (num_obs + 1, 1),
        dtype=torch.float32,
        requires_grad=False)

    num_dishes_poisson_rate_posteriors = torch.zeros(
        (num_obs + 1, 1),
        dtype=torch.float32,
        requires_grad=False)

    if likelihood_model == 'continuous_bernoulli':
        raise NotImplementedError
        # # need to use logits, otherwise gradient descent will carry parameters outside
        # # valid interval
        # likelihood_params = dict(
        #     logits=torch.full(
        #         size=(max_num_features, obs_dim),
        #         fill_value=np.nan,
        #         dtype=torch.float64,
        #         requires_grad=True)
        # )
        # create_new_feature_params_fn = create_new_cluster_params_continuous_bernoulli
        # posterior_fn = likelihood_continuous_bernoulli
        #
        # # make sure no observation is 0 or 1 by adding epsilon
        # epsilon = 1e-2
        # observations[observations == 1.] -= epsilon
        # observations[observations == 0.] += epsilon
    elif likelihood_model == 'dirichlet_multinomial':
        raise NotImplementedError
        # likelihood_params = dict(
        #     topics_concentrations=torch.full(
        #         size=(max_num_features, obs_dim),
        #         fill_value=np.nan,
        #         dtype=torch.float64,
        #         requires_grad=True),
        # )
        # create_new_feature_params_fn = create_new_cluster_params_dirichlet_multinomial
        # posterior_fn = likelihood_dirichlet_multinomial
    elif likelihood_model == 'multivariate_normal':
        if variable_variational_params is None:
            # we use half covariance because we want to numerically optimize
            A_half_covs = torch.stack([10. * torch.eye(obs_dim).float()
                                       for _ in range((num_obs + 1) * max_num_features)])
            A_half_covs = A_half_covs.view(num_obs + 1, max_num_features, obs_dim, obs_dim)

            # dict mapping variables to variational parameters
            variable_variational_params = dict(
                Z=dict(  # variational parameters for binary indicators
                    prob=torch.full(
                        size=(num_obs + 1, max_num_features),
                        fill_value=np.nan,
                        dtype=torch.float32),
                ),
                A=dict(  # variational parameters for Gaussian features
                    mean=torch.full(
                        size=(num_obs + 1, max_num_features, obs_dim),
                        fill_value=0.,
                        dtype=torch.float32),
                    # mean=A_mean,
                    half_cov=A_half_covs),
            )
    else:
        raise NotImplementedError

    # If we are optimizing numerically, set requires gradient to true, otherwise false
    for var_name, var_param_dict in variable_variational_params.items():
        for param_name, param_tensor in var_param_dict.items():
            param_tensor.requires_grad = numerically_optimize

    torch_observations = torch.from_numpy(observations).float()
    latent_indices = np.arange(max_num_features)

    # before the first observation, there are exactly 0 dishes
    num_dishes_poisson_rate_posteriors[0, 0] = 0.

    # REMEMBER: we added +1 to all the record-keeping arrays. Starting with 1
    # makes indexing consistent with the paper notation.
    for obs_idx, torch_observation in enumerate(torch_observations[:15], start=1):

        # construct number of dishes Poisson rate prior
        num_dishes_poisson_rate_priors[obs_idx, :] = num_dishes_poisson_rate_posteriors[obs_idx - 1, :] \
                                                     + alpha * beta / (beta + obs_idx - 1)

        # Recursion: 1st term
        dish_eating_prior = torch.clone(
            dish_eating_posteriors_running_sum[obs_idx - 1, :]) / (beta + obs_idx - 1)
        # Recursion: 2nd term; don't subtract 1 from latent indices b/c zero based indexing
        dish_eating_prior += poisson.cdf(k=latent_indices, mu=num_dishes_poisson_rate_posteriors[obs_idx - 1, :])
        # Recursion: 3rd term; don't subtract 1 from latent indices b/c zero based indexing
        dish_eating_prior -= poisson.cdf(k=latent_indices, mu=num_dishes_poisson_rate_priors[obs_idx, :])

        # record latent prior
        dish_eating_priors[obs_idx, :] = dish_eating_prior.clone()

        # initialize dish eating posterior to dish eating prior, before beginning inference
        variable_variational_params['Z']['prob'].data[obs_idx, :] = dish_eating_prior.clone()
        dish_eating_posteriors.data[obs_idx, :] = dish_eating_prior.clone()

        # initialize features to previous features as starting point for optimization
        # Use .data to not break backprop
        for param_name, param_tensor in variable_variational_params['A'].items():
            param_tensor.data[obs_idx, :] = param_tensor.data[obs_idx - 1, :]

        if plot_dir is not None:
            num_cols = 4
            fig, axes = plt.subplots(
                nrows=num_vi_steps_per_obs,
                ncols=num_cols,
                # sharex=True,
                # sharey=True,
                figsize=(num_cols * 4, num_vi_steps_per_obs * 4))

        approx_lower_bounds = []
        for vi_idx in range(num_vi_steps_per_obs):
            if numerically_optimize:
                # TODO: untested!
                # maximize approximate lower bound with respect to A's parameters
                approx_lower_bound = recursive_ibp_compute_approx_lower_bound(
                    torch_observation=torch_observation,
                    obs_idx=obs_idx,
                    dish_eating_prior=dish_eating_prior,
                    variable_variational_params=variable_variational_params)
                approx_lower_bounds.append(approx_lower_bound.item())
                approx_lower_bound.backward()

                # scale learning rate by posterior(A_k) / sum_n prev_posteriors(A_k)
                # scale by 1/num_vi_steps so that after num_vi_steps, we've moved O(1)
                scaled_learning_rate = learning_rate * torch.divide(
                    dish_eating_posteriors[obs_idx, :],
                    dish_eating_posteriors[obs_idx, :] + dish_eating_posteriors_running_sum[obs_idx - 1, :])
                scaled_learning_rate /= num_vi_steps_per_obs
                scaled_learning_rate[torch.isnan(scaled_learning_rate)] = 0.
                scaled_learning_rate[torch.isinf(scaled_learning_rate)] = 0.

                # make sure no gradient when applying gradient updates
                with torch.no_grad():
                    for var_name, var_dict in variable_variational_params.items():
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

            elif not numerically_optimize:
                with torch.no_grad():
                    logging.info(f'Obs Idx: {obs_idx}, VI idx: {vi_idx}')

                    tracemalloc.start()
                    start_time = timer()
                    approx_lower_bound = recursive_ibp_compute_approx_lower_bound(
                        torch_observation=torch_observation,
                        obs_idx=obs_idx,
                        dish_eating_prior=dish_eating_prior,
                        variable_variational_params=variable_variational_params)
                    approx_lower_bounds.append(approx_lower_bound.item())
                    stop_time = timer()
                    current, peak = tracemalloc.get_traced_memory()
                    logging.debug(f"memory:recursive_ibp_compute_approx_lower_bound:"
                                  f"current={current / 10 ** 6}MB; peak={peak / 10 ** 6}MB")
                    logging.debug(f'runtime:recursive_ibp_compute_approx_lower_bound: {stop_time - start_time}')
                    tracemalloc.stop()
                    # logging.info(prof.key_averages().table(sort_by="self_cpu_memory_usage"))

                    tracemalloc.start()
                    start_time = timer()
                    A_means, A_half_covs = recursive_ibp_optimize_gaussian_params(
                        torch_observation=torch_observation,
                        obs_idx=obs_idx,
                        vi_idx=vi_idx,
                        variable_variational_params=variable_variational_params,
                        simultaneous_or_sequential=simultaneous_or_sequential)
                    stop_time = timer()
                    current, peak = tracemalloc.get_traced_memory()
                    logging.debug(f"memory:recursive_ibp_optimize_gaussian_params:"
                                  f"current={current / 10 ** 6}MB; peak={peak / 10 ** 6}MB")
                    logging.debug(f'runtime:recursive_ibp_optimize_gaussian_params: {stop_time - start_time}')
                    tracemalloc.stop()

                    if ossify_features:
                        # TODO: refactor into own function
                        normalizing_const = torch.add(
                            variable_variational_params['Z']['prob'].data[obs_idx, :],
                            dish_eating_posteriors_running_sum[obs_idx - 1, :])
                        history_weighted_A_means = torch.add(
                            torch.multiply(variable_variational_params['Z']['prob'].data[obs_idx, :, None],
                                           A_means),
                            torch.multiply(dish_eating_posteriors_running_sum[obs_idx - 1, :, None],
                                           variable_variational_params['A']['mean'].data[obs_idx - 1, :]))
                        history_weighted_A_half_covs = torch.add(
                            torch.multiply(variable_variational_params['Z']['prob'].data[obs_idx, :, None, None],
                                           A_half_covs),
                            torch.multiply(dish_eating_posteriors_running_sum[obs_idx - 1, :, None, None],
                                           variable_variational_params['A']['half_cov'].data[obs_idx - 1, :]))
                        A_means = torch.divide(history_weighted_A_means, normalizing_const[:, None])
                        A_half_covs = torch.divide(history_weighted_A_half_covs, normalizing_const[:, None, None])
                        # if cumulative probability mass is 0, we compute 0/0 and get NaN. Need to mask
                        A_means[normalizing_const == 0.] = \
                            variable_variational_params['A']['mean'][obs_idx - 1][normalizing_const == 0.]
                        A_half_covs[normalizing_const == 0.] = \
                            variable_variational_params['A']['half_cov'][obs_idx - 1][normalizing_const == 0.]

                        utils.torch_helpers.assert_no_nan_no_inf(A_means)
                        utils.torch_helpers.assert_no_nan_no_inf(A_half_covs)

                    variable_variational_params['A']['mean'].data[obs_idx, :] = A_means
                    variable_variational_params['A']['half_cov'].data[obs_idx, :] = A_half_covs

                    # maximize approximate lower bound with respect to Z's parameters
                    tracemalloc.start()
                    start_time = timer()
                    Z_probs = recursive_ibp_optimize_bernoulli_params(
                        torch_observation=torch_observation,
                        obs_idx=obs_idx,
                        vi_idx=vi_idx,
                        dish_eating_prior=dish_eating_prior,
                        variable_variational_params=variable_variational_params,
                        simultaneous_or_sequential=simultaneous_or_sequential)
                    stop_time = timer()
                    current, peak = tracemalloc.get_traced_memory()
                    logging.debug(f"memory:recursive_ibp_optimize_bernoulli_params:"
                                  f"current={current / 10 ** 6}MB; peak={peak / 10 ** 6}MB")
                    logging.debug(f'runtime:recursive_ibp_optimize_bernoulli_params: {stop_time - start_time}')
                    tracemalloc.stop()

                    variable_variational_params['Z']['prob'].data[obs_idx, :] = Z_probs

                    # record dish-eating posterior
                    dish_eating_posteriors.data[obs_idx, :] = variable_variational_params['Z']['prob'][obs_idx, :].clone()

            else:
                raise ValueError(f'Impermissible value of numerically_optimize: {numerically_optimize}')

            if plot_dir is not None:
                fig.suptitle(f'Obs: {obs_idx}, {simultaneous_or_sequential}')
                axes[vi_idx, 0].set_ylabel(f'VI Step: {vi_idx + 1}')
                axes[vi_idx, 0].set_title('Individual Features')
                axes[vi_idx, 0].scatter(observations[:obs_idx, 0],
                                        observations[:obs_idx, 1],
                                        # s=3,
                                        color='k',
                                        label='Observations')
                for feature_idx in range(10):
                    axes[vi_idx, 0].plot(
                        [0, variable_variational_params['A']['mean'][obs_idx, feature_idx, 0].item()],
                        [0, variable_variational_params['A']['mean'][obs_idx, feature_idx, 1].item()],
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
                    variable_variational_params['A']['mean'][obs_idx, :, :].detach().numpy(),
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
                dish_eating_posteriors.data[obs_idx, :] = variable_variational_params['Z']['prob'][obs_idx, :].clone()

                # update running sum of posteriors
                dish_eating_posteriors_running_sum[obs_idx, :] = torch.add(
                    dish_eating_posteriors_running_sum[obs_idx - 1, :],
                    dish_eating_posteriors[obs_idx, :])

                # update how many dishes have been sampled
                non_eaten_dishes_posteriors_running_prod[obs_idx, :] = np.multiply(
                    non_eaten_dishes_posteriors_running_prod[obs_idx - 1, :],
                    1. - dish_eating_posteriors[obs_idx, :],  # p(z_{tk} = 0|o_{\leq t}) = 1 - p(z_{tk} = 1|o_{\leq t})
                )

                approx_lower_bound = recursive_ibp_compute_approx_lower_bound(
                    torch_observation=torch_observation,
                    obs_idx=obs_idx,
                    dish_eating_prior=dish_eating_prior,
                    variable_variational_params=variable_variational_params)
                approx_lower_bounds.append(approx_lower_bound.item())

                num_dishes_poisson_rate_posteriors[obs_idx] = torch.sum(
                    1. - non_eaten_dishes_posteriors_running_prod[obs_idx, :])

                # update running sum of which customers ate which dishes
                dish_eating_posteriors_running_sum[obs_idx] = torch.add(
                    dish_eating_posteriors_running_sum[obs_idx - 1, :],
                    dish_eating_posteriors[obs_idx, :])

        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir, f'{simultaneous_or_sequential}_params_obs={obs_idx}.png'),
                        bbox_inches='tight',
                        dpi=300)
            # plt.show()
            plt.close()

            plt.scatter(1 + np.arange(len(approx_lower_bounds)),
                        approx_lower_bounds)
            plt.xlabel('VI Step')
            plt.ylabel('VI Approx Lower Bound')
            plt.savefig(os.path.join(plot_dir, f'{simultaneous_or_sequential}_approxlowerbound_obs={obs_idx}.png'),
                        bbox_inches='tight',
                        dpi=300)
            # plt.show()
            plt.close()

    # Remember to cut off the first index.y
    numpy_variable_variational_params = {
        var_name: {param_name: param_tensor.detach().numpy()[1:]
                   for param_name, param_tensor in var_params.items()}
        for var_name, var_params in variable_variational_params.items()
    }
    bayesian_recursion_results = dict(
        dish_eating_priors=dish_eating_priors.numpy()[1:],
        dish_eating_posteriors=dish_eating_posteriors.numpy()[1:],
        dish_eating_posteriors_running_sum=dish_eating_posteriors_running_sum.numpy()[1:],
        non_eaten_dishes_posteriors_running_prod=non_eaten_dishes_posteriors_running_prod.numpy()[1:],
        num_dishes_poisson_rate_priors=num_dishes_poisson_rate_priors.numpy()[1:],
        num_dishes_poisson_rate_posteriors=num_dishes_poisson_rate_posteriors.numpy()[1:],
        variable_variational_params=numpy_variable_variational_params,
    )

    return bayesian_recursion_results


def recursive_ibp_compute_approx_lower_bound(torch_observation,
                                             obs_idx,
                                             dish_eating_prior,
                                             variable_variational_params):
    logging.debug('entering:recursive_ibp_compute_approx_lower_bound')
    # convert half covariances to covariances
    prior_A_cov = utils.torch_helpers.convert_half_cov_to_cov(
        half_cov=variable_variational_params['A']['half_cov'][obs_idx - 1])
    posterior_A_cov = utils.torch_helpers.convert_half_cov_to_cov(
        half_cov=variable_variational_params['A']['half_cov'][obs_idx])

    indicators_term = utils.torch_helpers.expected_log_bernoulli_under_bernoulli(
        p_prob=dish_eating_prior,
        q_prob=variable_variational_params['Z']['prob'][obs_idx])
    gaussian_term = utils.torch_helpers.expected_log_gaussian_under_gaussian(
        p_mean=variable_variational_params['A']['mean'][obs_idx - 1],
        p_cov=prior_A_cov,
        q_mean=variable_variational_params['A']['mean'][obs_idx],
        q_cov=posterior_A_cov)
    likelihood_term = utils.torch_helpers.expected_log_gaussian_under_linear_gaussian(
        observation=torch_observation,
        q_A_mean=variable_variational_params['A']['mean'][obs_idx],
        q_A_cov=posterior_A_cov,
        q_Z_mean=variable_variational_params['Z']['prob'][obs_idx])
    bernoulli_entropy = utils.torch_helpers.entropy_bernoulli(
        probs=variable_variational_params['Z']['prob'][obs_idx])
    gaussian_entropy = utils.torch_helpers.entropy_gaussian(
        mean=variable_variational_params['A']['mean'][obs_idx],
        cov=posterior_A_cov)

    lower_bound = indicators_term + gaussian_term + likelihood_term + bernoulli_entropy + gaussian_entropy

    utils.torch_helpers.assert_no_nan_no_inf(lower_bound)
    logging.debug('exiting:recursive_ibp_compute_approx_lower_bound')
    return lower_bound


def recursive_ibp_optimize_bernoulli_params(torch_observation: torch.Tensor,
                                            obs_idx: int,
                                            vi_idx: int,
                                            dish_eating_prior: torch.Tensor,
                                            variable_variational_params: Dict[str, dict],
                                            sigma_obs_squared: int = 1e-0,
                                            simultaneous_or_sequential: str = 'sequential') -> torch.Tensor:
    assert simultaneous_or_sequential in {'sequential', 'simultaneous'}

    A_cov = utils.torch_helpers.convert_half_cov_to_cov(
        variable_variational_params['A']['half_cov'][obs_idx, :])

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
        log_bernoulli_prior_term = torch.log(
            torch.divide(dish_eating_prior[slice_idx],
                         1. - dish_eating_prior[slice_idx]))

        # -2 mu_{nk}^T o_n
        term_one = -2. * torch.einsum(
            'bk,k->b',
            variable_variational_params['A']['mean'][obs_idx, slice_idx],
            torch_observation)

        # Tr[\Sigma_{nk} + \mu_{nk} \mu_{nk}^T]
        term_two = torch.einsum(
            'bii->b',
            torch.add(A_cov[slice_idx],
                      torch.einsum('bi,bj->bij',
                                   variable_variational_params['A']['mean'][obs_idx, slice_idx],
                                   variable_variational_params['A']['mean'][obs_idx, slice_idx])))

        # \mu_{nk}^T (\sum_{k': k' \neq k} b_{nk'} \mu_{nk'})
        # = \mu_{nk}^T (\sum_{k'} b_{nk'} \mu_{nk'}) - b_{nk} \mu_{nk}^T \mu_{nk}
        term_three_all_pairs = torch.einsum(
            'bi, i->b',
            variable_variational_params['A']['mean'][obs_idx, slice_idx],
            torch.einsum(
                'b, bi->i',
                bernoulli_probs,
                variable_variational_params['A']['mean'][obs_idx, :]))
        term_three_self_pairs = torch.einsum(
            'b,bk,bk->b',
            bernoulli_probs[slice_idx],
            variable_variational_params['A']['mean'][obs_idx, slice_idx],
            variable_variational_params['A']['mean'][obs_idx, slice_idx])
        term_three = term_three_all_pairs - term_three_self_pairs

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


def run_inference_alg(inference_alg_str: str,
                      observations: np.ndarray,
                      inference_alg_params: dict,
                      likelihood_model: str,
                      plot_dir: str = None,
                      learning_rate: float = 1e0):

    # create dict to store algorithm-specific arguments to inference alg function
    inference_alg_kwargs = dict()

    # select inference alg and add kwargs as necessary
    if inference_alg_str == 'R-IBP':
        inference_alg_fn = recursive_ibp
        # TODO: add inference alg kwargs for which likelihood to use
    # elif inference_alg_str == 'Online CRP':
    #     inference_alg_fn = online_crp
    # elif inference_alg_str == 'SUSG':
    #     inference_alg_fn = sequential_updating_and_greedy_search
    # elif inference_alg_str == 'VSUSG':
    #     inference_alg_fn = variational_sequential_updating_and_greedy_search
    # elif inference_alg_str.startswith('DP-Means'):
    #     inference_alg_fn = dp_means
    #     if inference_alg_str.endswith('(offline)'):
    #         inference_alg_kwargs['num_passes'] = 8  # same as Kulis and Jordan
    #     elif inference_alg_str.endswith('(online)'):
    #         inference_alg_kwargs['num_passes'] = 1
    #     else:
    #         raise ValueError('Invalid DP Means')
    # elif inference_alg_str.startswith('HMC-Gibbs'):
    #     #     gaussian_cov_scaling=gaussian_cov_scaling,
    #     #     gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling,
    #     inference_alg_fn = sampling_hmc_gibbs
    #
    #     # Suppose inference_alg_str is 'HMC-Gibbs (5000 Samples)'. We want to extract
    #     # the number of samples. To do this, we use the following
    #     num_samples = int(inference_alg_str.split(' ')[1][1:])
    #     inference_alg_kwargs['num_samples'] = num_samples
    #     inference_alg_kwargs['truncation_num_clusters'] = 12
    #
    #     if likelihood_model == 'dirichlet_multinomial':
    #         inference_alg_kwargs['model_params'] = dict(
    #             dirichlet_inference_params=10.)  # same as R-CRP
    #     elif likelihood_model == 'multivariate_normal':
    #         # Note: these are the ground truth parameters
    #         inference_alg_kwargs['model_params'] = dict(
    #             gaussian_mean_prior_cov_scaling=6,
    #             gaussian_cov_scaling=0.3)
    #     else:
    #         raise ValueError(f'Unknown likelihood model: {likelihood_model}')
    # elif inference_alg_str.startswith('SVI'):
    #     inference_alg_fn = stochastic_variational_inference
    #     learning_rate = 5e-4
    #     # suppose the inference_alg_str is 'SVI (5k Steps)'
    #     substrs = inference_alg_str.split(' ')
    #     num_steps = 1000 * int(substrs[1][1:-1])
    #     inference_alg_kwargs['num_steps'] = num_steps
    #     # Note: these are the ground truth parameters
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
    #     inference_alg_fn = variational_bayes
    #     # Suppose we have an algorithm string 'Variational Bayes (10 Init, 10 Iterations)',
    #     substrs = inference_alg_str.split(' ')
    #     num_initializations = int(substrs[2][1:])
    #     max_iters = int(substrs[4])
    #     inference_alg_kwargs['num_initializations'] = num_initializations
    #     inference_alg_kwargs['max_iter'] = max_iters
    else:
        raise ValueError(f'Unknown inference algorithm: {inference_alg_str}')

    # run inference algorithm
    inference_alg_results = inference_alg_fn(
        observations=observations,
        inference_params=inference_alg_params,
        likelihood_model=likelihood_model,
        learning_rate=learning_rate,
        plot_dir=plot_dir,
        **inference_alg_kwargs)

    return inference_alg_results
