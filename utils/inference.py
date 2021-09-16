import cvxpy as cp
import jax
import jax.numpy as jnp
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
from typing import Dict, Tuple

import utils.torch_helpers
import utils.inference_widjaja
import utils.inference_refactor

# torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_default_tensor_type('torch.FloatTensor')

inference_alg_strs = [
    'HMC-Gibbs',
    'Doshi-Velez',
    'R-IBP',
    'Widjaja',
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
                                             variable_variational_params: Dict[str, torch.Tensor],
                                             sigma_obs_squared: float):
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
        q_Z_mean=variable_variational_params['Z']['prob'][obs_idx],
        sigma_obs_squared=sigma_obs_squared)
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
                      plot_dir: str = None,
                      learning_rate: float = 1e0):

    # create dict to store algorithm-specific arguments to inference alg function
    if model_params is None:
        model_params = {}

    if model_str == 'linear_gaussian':
        model_params['gaussian_likelihood_cov_scaling'] = 1.
        model_params['gaussian_prior_cov_scaling']  = 100.
    elif model_str == 'factor_analysis':
        raise NotImplementedError
    elif model_str == 'nonnegative_matrix_factorization':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # select inference alg and add kwargs as necessary
    if inference_alg_str == 'Doshi-Velez':
        inference_alg = variational_inference_offline
    elif inference_alg_str == 'R-IBP':
        inference_alg = utils.inference_refactor.RecursiveIBP(
            model_str=model_str,
            model_params=model_params,
            plot_dir=plot_dir)
    elif inference_alg_str == 'Widjaja':
        inference_alg = variational_inference_online
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
        inference_alg = utils.inference_refactor.HMCGibbs(
            model_str=model_str,
            model_params=model_params,
            num_samples=100,
            num_warmup_samples=15,
            num_thinning_samples=10)

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
    #     # Note: these are the ground truth parameters
    #     if likelihood_model == 'dirichlet_multinomial':
    #         inference_alg_kwargs['model_parameters'] = dict(
    #             dirichlet_inference_params=10.)  # same as R-CRP
    #     elif likelihood_model == 'multivariate_normal':
    #         inference_alg_kwargs['model_parameters'] = dict(
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


def variational_inference_offline(observations,
                                  inference_params: Dict[str, float],
                                  likelihood_model: str,
                                  model_parameters: Dict[str, float],
                                  max_num_features: int = None,
                                  num_coordinate_ascent_steps: int = 100,
                                  plot_dir: str = None, ):
    """
    Implementation of Doshi-Velez 2009 Variational Inference for the IBP.
    """

    if likelihood_model != 'linear_gaussian':
        raise NotImplementedError

    assert num_coordinate_ascent_steps > 0
    num_obs, obs_dim = observations.shape
    if max_num_features is None:
        # Note: the expected number of latents grows logarithmically as a*b*log(1 + N/sticks)
        # The 10 is a hopefully conservative heuristic to preallocate.
        max_num_features = 10 * int(inference_params['alpha'] * inference_params['beta'] * \
                                    np.log(1 + num_obs / inference_params['beta']))

    offline_model = utils.inference_widjaja.OfflineFinite(
        obs_dim=obs_dim,
        num_obs=num_obs,
        max_num_features=max_num_features,
        alpha=inference_params['alpha'],
        beta=inference_params['beta'],
        sigma_a=np.sqrt(model_parameters['gaussian_prior_cov_scaling']),
        sigma_x=np.sqrt(model_parameters['gaussian_likelihood_cov_scaling']),
        t0=1,
        kappa=0.5)

    offline_strategy = utils.inference_widjaja.Static(
        offline_model,
        observations,
        minibatch_size=num_obs,  # full batch
    )

    dish_eating_priors = np.zeros(shape=(num_obs, max_num_features))
    dish_eating_posteriors = np.zeros(shape=(num_obs, max_num_features))
    beta_param_1 = np.zeros(shape=(max_num_features,))
    beta_param_2 = np.zeros(shape=(max_num_features,))

    # Always use full batch
    obs_indices = slice(0, num_obs, 1)
    for step_idx in range(num_coordinate_ascent_steps):
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
    variable_parameters = dict(
        A=dict(mean=A_means, cov=A_covs),
        pi=dict(param_1=beta_param_1, param_2=beta_param_2)
    )

    dish_eating_posteriors_running_sum = np.cumsum(dish_eating_posteriors, axis=0)

    num_dishes_poisson_rate_posteriors = np.sum(
        dish_eating_posteriors_running_sum > 1e-10,
        axis=1).reshape(-1, 1)

    num_dishes_poisson_rate_priors = np.full(fill_value=np.nan,
                                             shape=num_dishes_poisson_rate_posteriors.shape)

    variational_inference_offline_results = dict(
        dish_eating_priors=dish_eating_priors,
        dish_eating_posteriors=dish_eating_posteriors,
        dish_eating_posteriors_running_sum=dish_eating_posteriors_running_sum,
        num_dishes_poisson_rate_priors=num_dishes_poisson_rate_priors,
        num_dishes_poisson_rate_posteriors=num_dishes_poisson_rate_posteriors,
        variable_parameters=variable_parameters,
        model_parameters=model_parameters,
    )

    return variational_inference_offline_results


def variational_inference_online(observations,
                                 inference_params: Dict[str, float],
                                 likelihood_model: str,
                                 model_parameters: Dict[str, float],
                                 max_num_features: int = None,
                                 plot_dir: str = None, ):
    """
    Implementation of Widjaja 2017 Streaming Variational Inference for the IBP.
    """

    if likelihood_model != 'linear_gaussian':
        raise NotImplementedError

    num_obs, obs_dim = observations.shape
    if max_num_features is None:
        # Note: the expected number of latents grows logarithmically as a*b*log(1 + N/sticks)
        # The 10 is a hopefully conservative heuristic to preallocate.
        max_num_features = 10 * int(inference_params['alpha'] * inference_params['beta'] * \
                                    np.log(1 + num_obs / inference_params['beta']))

    online_model = utils.inference_widjaja.OnlineFinite(
        obs_dim=obs_dim,
        max_num_features=max_num_features,
        alpha=inference_params['alpha'],
        beta=inference_params['beta'],
        sigma_a=np.sqrt(model_parameters['gaussian_prior_cov_scaling']),
        sigma_x=np.sqrt(model_parameters['gaussian_likelihood_cov_scaling']),
        t0=1,
        kappa=0.5)

    online_strategy = utils.inference_widjaja.Static(
        online_model,
        observations,
        minibatch_size=10)

    dish_eating_priors = np.zeros(shape=(num_obs, max_num_features))

    dish_eating_posteriors = np.zeros(shape=(num_obs, max_num_features))

    beta_param_1 = np.zeros(shape=(num_obs, max_num_features))
    beta_param_2 = np.zeros(shape=(num_obs, max_num_features))

    A_means = np.zeros(shape=(num_obs, max_num_features, obs_dim))
    # They assume a diagonal covariance. We will later expand.
    A_covs = np.zeros(shape=(num_obs, max_num_features, obs_dim))

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
    variable_parameters = dict(
        A=dict(mean=A_means, cov=A_covs),
        pi=dict(param_1=beta_param_1, param_2=beta_param_2)
    )

    dish_eating_posteriors_running_sum = np.cumsum(dish_eating_posteriors, axis=0)

    num_dishes_poisson_rate_posteriors = np.sum(dish_eating_posteriors_running_sum > 1e-2,
                                                axis=1).reshape(-1, 1)

    dish_eating_priors_running_sum = np.cumsum(dish_eating_priors, axis=0)
    num_dishes_poisson_rate_priors = np.sum(dish_eating_priors_running_sum > 1e-2,
                                            axis=1).reshape(-1, 1)

    variational_inference_online_results = dict(
        dish_eating_priors=dish_eating_priors,
        dish_eating_posteriors=dish_eating_posteriors,
        dish_eating_posteriors_running_sum=dish_eating_posteriors_running_sum,
        num_dishes_poisson_rate_priors=num_dishes_poisson_rate_priors,
        num_dishes_poisson_rate_posteriors=num_dishes_poisson_rate_posteriors,
        variable_parameters=variable_parameters,
        model_parameters=model_parameters,
    )

    return variational_inference_online_results
