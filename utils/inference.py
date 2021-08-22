import cvxpy as cp
import numpy as np
from scipy.stats import poisson
import torch
from typing import Dict

from utils.helpers import assert_torch_no_nan_no_inf, torch_logits_to_probs, torch_probs_to_logits

torch.set_default_tensor_type('torch.DoubleTensor')

inference_alg_strs = [
    'R-IBP',
]


def create_new_feature_params_multivariate_normal(torch_observation: torch.Tensor,
                                                  dish_eating_prior: torch.Tensor,
                                                  obs_idx: int,
                                                  likelihood_params: Dict[str, torch.Tensor]):
    # data is necessary to not break backprop
    # see https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
    assert_torch_no_nan_no_inf(torch_observation)
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
                  likelihood_known: bool = False,
                  likelihood_params: dict = None,
                  learning_rate: float = 1e0,
                  num_em_steps: int = 3):
    for param, param_value in inference_params.items():
        assert param_value > 0
    alpha = inference_params['alpha']
    beta = inference_params['beta']

    assert likelihood_model in {'multivariate_normal', 'dirichlet_multinomial',
                                'bernoulli', 'continuous_bernoulli', 'linear_gaussian'}

    num_obs, obs_dim = observations.shape

    # Note: the expected number of latents grows logarithmically as a*b*log(b + T - 1)
    # The 10 is a hopefully conservative heuristic to preallocate.
    max_num_features = 10 * int(inference_params['alpha'] * inference_params['beta'] * \
                                np.log(inference_params['beta'] + num_obs - 1))

    # The recursion does not require recording the full history of priors/posteriors
    # but we record the full history for subsequent analysis.
    dish_eating_priors = torch.zeros(
        (num_obs + 1, max_num_features),  # Add 1 to the number of observations to use 1-based indexing
        dtype=torch.float64,
        requires_grad=False)

    dish_eating_posteriors = torch.zeros(
        (num_obs + 1, max_num_features),
        dtype=torch.float64,
        requires_grad=False)

    dish_eating_posteriors_running_sum = torch.zeros(
        (num_obs + 1, max_num_features),
        dtype=torch.float64,
        requires_grad=False)

    non_eaten_dishes_posteriors_running_prod = torch.ones(
        (num_obs + 1, max_num_features),
        dtype=torch.float64,
        requires_grad=False)

    num_dishes_poisson_rate_priors = torch.zeros(
        (num_obs + 1, 1),
        dtype=torch.float64,
        requires_grad=False)

    num_dishes_poisson_rate_posteriors = torch.zeros(
        (num_obs + 1, 1),
        dtype=torch.float64,
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
        # if likelihood_known:
        #     assert likelihood_params['mean'].shape == (max_num_features, obs_dim)
        #     assert likelihood_params['cov'].shape == (obs_dim, obs_dim)
        #     # torch.from_numpy implicitly sets requires_grad to False
        #     likelihood_params = dict(
        #         mean=torch.from_numpy(likelihood_params['mean']),
        #         cov=torch.from_numpy(likelihood_params['cov'])
        #     )
        # else:
        likelihood_params = dict(
            means=torch.full(
                size=(max_num_features, obs_dim),
                fill_value=0.,
                dtype=torch.float64,
                requires_grad=True),
            cov=torch.full(
                size=(obs_dim, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
        )
        create_new_feature_params_fn = create_new_feature_params_multivariate_normal
        # posterior_fn = posterior_multivariate_normal_linear_regression_leave_one_out
        # posterior_fn = posterior_multivariate_normal_linear_regression_forward_stepwise
        posterior_fn = posterior_multivariate_normal_linear_regression_simultaneous
    else:
        raise NotImplementedError

    optimizer = torch.optim.SGD(params=likelihood_params.values(), lr=1.)

    torch_observations = torch.from_numpy(observations)
    latent_indices = np.arange(max_num_features)

    # before the first observation, there are exactly 0 dishes
    num_dishes_poisson_rate_posteriors[0, 0] = 0.

    # REMEMBER: we added +1 to all the record-keeping arrays. Starting with 1
    # makes indexing consistent with the paper notation.
    for obs_idx, torch_observation in enumerate(torch_observations, start=1):

        # construct priors
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

        # create new params for possible cluster, centered at that point
        create_new_feature_params_fn(
            dish_eating_prior=dish_eating_prior,
            torch_observation=torch_observation,
            obs_idx=obs_idx,
            likelihood_params=likelihood_params)

        for em_idx in range(num_em_steps):

            optimizer.zero_grad()

            # E step: infer posteriors using parameters
            posterior_fn(
                torch_observation=torch_observation,
                likelihood_params=likelihood_params)
            assert torch.all(~torch.isnan(likelihoods_per_latent[:obs_idx + 1]))
            assert torch.all(~torch.isnan(log_likelihoods_per_latent[:obs_idx + 1]))

            unnormalized_table_assignment_posterior = torch.multiply(
                likelihoods_per_latent.detach(),
                dish_eating_prior)
            table_assignment_posterior = unnormalized_table_assignment_posterior / torch.sum(
                unnormalized_table_assignment_posterior)
            assert torch.allclose(torch.sum(table_assignment_posterior), one_tensor)

            # record latent posterior
            dish_eating_posteriors[obs_idx, :len(table_assignment_posterior)] = table_assignment_posterior

            # update running sum of posteriors
            dish_eating_posteriors_running_sum[obs_idx, :] = torch.add(
                dish_eating_posteriors_running_sum[obs_idx - 1, :],
                dish_eating_posteriors[obs_idx, :])
            assert torch.allclose(torch.sum(dish_eating_posteriors_running_sum[obs_idx, :]),
                                  torch.Tensor([obs_idx + 1]).double())

            # M step: update parameters
            # Note: log likelihood is all we need for optimization because
            # log p(x, z; params) = log p(x|z; params) + log p(z)
            # and the second is constant w.r.t. params gradient
            loss = torch.mean(log_likelihoods_per_latent)
            loss.backward()

            # instead of typical dynamics:
            #       p_k <- p_k + (obs - p_k) / number of obs assigned to kth cluster
            # we use the new dynamics
            #       p_k <- p_k + posterior(obs belongs to kth cluster) * (obs - p_k) / total mass on kth cluster
            # that effectively means the learning rate should be this scaled_prefactor
            scaled_learning_rate = learning_rate * torch.divide(
                dish_eating_posteriors[obs_idx, :],
                dish_eating_posteriors_running_sum[obs_idx, :]) / num_em_steps
            scaled_learning_rate[torch.isnan(scaled_learning_rate)] = 0.
            scaled_learning_rate[torch.isinf(scaled_learning_rate)] = 0.

            # don't update the newest cluster
            scaled_learning_rate[obs_idx] = 0.

            for param_descr, param_tensor in likelihood_params.items():
                # the scaled learning rate has shape (num latents,) aka (num obs,)
                # we need to create extra dimensions of size 1 for broadcasting to work
                # because param_tensor can have variable number of dimensions e.g. (num obs, obs dim)
                # for mean vs (num obs, obs dim, obs dim) for covariance, we need to dynamically
                # add the corect number of dimensions
                reshaped_scaled_learning_rate = scaled_learning_rate.view(
                    [scaled_learning_rate.shape[0]] + [1 for _ in range(len(param_tensor.shape[1:]))])
                if param_tensor.grad is None:
                    continue
                else:
                    scaled_param_tensor_grad = torch.multiply(
                        reshaped_scaled_learning_rate,
                        param_tensor.grad)
                    param_tensor.data += scaled_param_tensor_grad
                    assert_torch_no_nan_no_inf(param_tensor.data[:obs_idx + 1])

        # update how many dishes have been sampled
        non_eaten_dishes_posteriors_running_prod[obs_idx, :] = np.multiply(
            non_eaten_dishes_posteriors_running_prod[obs_idx - 1, :],
            1 - dish_eating_posterior,  # p(z_{tk} = 0|o_{\leq t}) = 1 - p(z_{tk} = 1|o_{\leq t})
        )

        num_dishes_poisson_rate_posteriors[obs_idx] = torch.sum(
            1 - non_eaten_dishes_posteriors_running_prod[obs_idx, :])

        # update running sum of which customers ate which dishes
        dish_eating_posteriors_running_sum[obs_idx] = torch.add(
            dish_eating_posteriors_running_sum[obs_idx - 1, :],
            dish_eating_posteriors[obs_idx, :])

    bayesian_recursion_results = dict(
        dish_eating_priors=dish_eating_priors.numpy(),
        dish_eating_posteriors=dish_eating_posteriors.numpy(),
        dish_eating_posteriors_running_sum=dish_eating_posteriors_running_sum.numpy(),
        non_eaten_dishes_posteriors_running_prod=non_eaten_dishes_posteriors_running_prod.numpy(),
        num_dishes_poisson_rate_priors=num_dishes_poisson_rate_priors.numpy(),
        num_dishes_poisson_rate_posteriors=num_dishes_poisson_rate_posteriors.numpy(),
        parameters={k: v.detach().numpy() for k, v in likelihood_params.items()},
    )

    return bayesian_recursion_results


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
        print(1)

    return dish_eating_posterior


def run_inference_alg(inference_alg_str: str,
                      observations: np.ndarray,
                      inference_alg_params: dict,
                      likelihood_model: str,
                      learning_rate: float = 1e0):
    # if likelihood_known:
    #     assert likelihood_params is not None

    # allow algorithm-specific arguments to inference alg function
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
        **inference_alg_kwargs)

    return inference_alg_results
