import numpy as np
import torch

from utils.helpers import assert_torch_no_nan_no_inf, torch_logits_to_probs, torch_probs_to_logits

torch.set_default_tensor_type('torch.DoubleTensor')


def create_new_cluster_params_linear_gaussian(torch_observation,
                                              obs_idx,
                                              likelihood_params):
    # data is necessary to not break backprop
    # see https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
    assert_torch_no_nan_no_inf(torch_observation)
    likelihood_params['M'].data[obs_idx, :] = torch_observation
    likelihood_params['stddevs'].data[obs_idx, :, :] = torch.eye(torch_observation.shape[0])


def create_new_feature_params_multivariate_normal(torch_observation,
                                                  obs_idx,
                                                  likelihood_params):
    # data is necessary to not break backprop
    # see https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
    assert_torch_no_nan_no_inf(torch_observation)
    likelihood_params['means'].data[obs_idx, :] = torch_observation
    likelihood_params['stddevs'].data[obs_idx, :, :] = torch.eye(torch_observation.shape[0])


def recursive_ibp(observations,
                  inference_params: dict,
                  likelihood_model: str,
                  likelihood_known: bool = False,
                  likelihood_params: dict = None,
                  learning_rate: float = 1e0,
                  num_em_steps: int = 3):

    for param, param_value in inference_params.items():
        assert param_value > 0

    assert likelihood_model in {'multivariate_normal', 'dirichlet_multinomial',
                                'bernoulli', 'continuous_bernoulli', 'linear_gaussian'}
    num_obs, obs_dim = observations.shape

    if likelihood_known:
        assert likelihood_params is not None
        num_em_steps = 0  # don't need to update likelihood parameters if known!
        # first out max number of dishes, based on first (arbitrary) key
        param_str = list(likelihood_params.keys())[0]
        max_num_latents = likelihood_params[param_str].shape[0]
    else:
        # Note: the expected number of latents grows logarithmically as a*b*log(b + T - 1)
        # The 10 is a heuristic to be safe.
        max_num_latents = int(10 * inference_params['alpha'] * inference_params['beta'] * \
                              np.log(inference_params['beta'] + num_obs - 1))

    # The recursion does not require recording the full history of priors/posteriors
    # but we record the full history for subsequent analysis
    dish_eating_priors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    dish_eating_posteriors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    dish_eating_posteriors_running_sum = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    non_eaten_dishes_posteriors_running_prod = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    num_dishes_poisson_rate_posteriors = torch.zeros(
        (num_obs, 1),
        dtype=torch.float64,
        requires_grad=False)

    if likelihood_model == 'continuous_bernoulli':
        raise NotImplementedError
        # # need to use logits, otherwise gradient descent will carry parameters outside
        # # valid interval
        # likelihood_params = dict(
        #     logits=torch.full(
        #         size=(max_num_latents, obs_dim),
        #         fill_value=np.nan,
        #         dtype=torch.float64,
        #         requires_grad=True)
        # )
        # create_new_feature_params_fn = create_new_cluster_params_continuous_bernoulli
        # likelihood_fn = likelihood_continuous_bernoulli
        #
        # # make sure no observation is 0 or 1 by adding epsilon
        # epsilon = 1e-2
        # observations[observations == 1.] -= epsilon
        # observations[observations == 0.] += epsilon
    elif likelihood_model == 'dirichlet_multinomial':
        raise NotImplementedError
        # likelihood_params = dict(
        #     topics_concentrations=torch.full(
        #         size=(max_num_latents, obs_dim),
        #         fill_value=np.nan,
        #         dtype=torch.float64,
        #         requires_grad=True),
        # )
        # create_new_feature_params_fn = create_new_cluster_params_dirichlet_multinomial
        # likelihood_fn = likelihood_dirichlet_multinomial
    elif likelihood_model == 'multivariate_normal':
        if likelihood_known:
            assert likelihood_params['mean'].shape == (max_num_latents, obs_dim)
            assert likelihood_params['cov'].shape == (obs_dim, obs_dim)
            # torch.from_numpy implicitly sets requires_grad to False
            likelihood_params = dict(
                mean=torch.from_numpy(likelihood_params['mean']),
                cov=torch.from_numpy(likelihood_params['cov'])
            )
        else:
            likelihood_params = dict(
                means=torch.full(
                    size=(max_num_latents, obs_dim),
                    fill_value=np.nan,
                    dtype=torch.float64,
                    requires_grad=True),
                cov=torch.full(
                    size=(obs_dim, obs_dim),
                    fill_value=np.nan,
                    dtype=torch.float64,
                    requires_grad=True),
            )
        create_new_feature_params_fn = create_new_feature_params_multivariate_normal
        likelihood_fn = likelihood_multivariate_normal
    else:
        raise NotImplementedError

    # optimizer = torch.optim.SGD(params=[logits], lr=1.)
    optimizer = torch.optim.SGD(params=likelihood_params.values(), lr=1.)

    # used for later checking that vectors sum to 1.
    one_tensor = torch.Tensor([1.]).double()

    torch_observations = torch.from_numpy(observations)
    for obs_idx, torch_observation in enumerate(torch_observations):

        if not likelihood_known:
            # create new params for possible cluster, centered at that point
            create_new_feature_params_fn(
                torch_observation=torch_observation,
                obs_idx=obs_idx,
                likelihood_params=likelihood_params)

        if obs_idx == 0:
            # first customer has to go at first table
            dish_eating_priors[obs_idx, 0] = 1.
            dish_eating_posteriors[obs_idx, 0] = 1.
            num_dishes_poisson_rate_posteriors[obs_idx, 0] = 1.

            # update running sum of posteriors
            dish_eating_posteriors_running_sum[obs_idx, :] = torch.add(
                dish_eating_posteriors_running_sum[obs_idx - 1, :],
                dish_eating_posteriors[obs_idx, :])
            assert torch.allclose(torch.sum(dish_eating_posteriors_running_sum[obs_idx, :]),
                                  torch.Tensor([obs_idx + 1]).double())
        else:
            # construct prior
            table_assignment_prior = torch.clone(
                dish_eating_posteriors_running_sum[obs_idx - 1, :obs_idx + 1])
            # we don't subtract 1 because Python uses 0-based indexing
            assert torch.allclose(torch.sum(table_assignment_prior), torch.Tensor([obs_idx]).double())
            # add new table probability
            table_assignment_prior[1:] += inference_params * torch.clone(
                num_dishes_poisson_rate_posteriors[obs_idx - 1, :obs_idx])
            # renormalize
            table_assignment_prior /= (inference_params + obs_idx)
            assert torch.allclose(torch.sum(table_assignment_prior), one_tensor)

            # record latent prior
            dish_eating_priors[obs_idx, :len(table_assignment_prior)] = table_assignment_prior

            for em_idx in range(num_em_steps):

                optimizer.zero_grad()

                # E step: infer posteriors using parameters
                likelihoods_per_latent, log_likelihoods_per_latent = likelihood_fn(
                    torch_observation=torch_observation,
                    obs_idx=obs_idx,
                    likelihood_params=likelihood_params)
                assert torch.all(~torch.isnan(likelihoods_per_latent[:obs_idx + 1]))
                assert torch.all(~torch.isnan(log_likelihoods_per_latent[:obs_idx + 1]))

                unnormalized_table_assignment_posterior = torch.multiply(
                    likelihoods_per_latent.detach(),
                    table_assignment_prior)
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

            # # previous approach with time complexity O(t^2)
            # # update posterior over number of tables using posterior over customer seat
            # for k1, p_z_t_equals_k1 in enumerate(dish_eating_posteriors[obs_idx, :obs_idx + 1]):
            #     for k2, p_prev_num_tables_equals_k2 in enumerate(num_dishes_poisson_rate_posteriors[obs_idx - 1, :obs_idx + 1]):
            #         # advance number of tables by 1 if customer seating > number of current tables
            #         if k1 > k2 + 1:
            #             num_dishes_poisson_rate_posteriors.data[obs_idx, k2 + 1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
            #         # customer allocated to previous table
            #         elif k1 <= k2:
            #             num_dishes_poisson_rate_posteriors.data[obs_idx, k2] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
            #         # create new table
            #         elif k1 == k2 + 1:
            #             num_dishes_poisson_rate_posteriors.data[obs_idx, k1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
            #         else:
            #             raise ValueError
            # assert torch.allclose(torch.sum(num_dishes_poisson_rate_posteriors[obs_idx, :]), one_tensor)

            # new approach with time complexity O(t)
            # update posterior over number of tables using posterior over customer seat
            cum_table_assignment_posterior = torch.cumsum(
                dish_eating_posteriors[obs_idx, :obs_idx + 1],
                dim=0)
            one_minus_cum_table_assignment_posterior = 1. - cum_table_assignment_posterior
            prev_table_posterior = num_dishes_poisson_rate_posteriors[obs_idx - 1, :obs_idx]
            num_dishes_poisson_rate_posteriors[obs_idx, :obs_idx] += torch.multiply(
                cum_table_assignment_posterior[:-1],
                prev_table_posterior)
            num_dishes_poisson_rate_posteriors[obs_idx, 1:obs_idx + 1] += torch.multiply(
                one_minus_cum_table_assignment_posterior[:-1],
                prev_table_posterior)
            assert torch.allclose(torch.sum(num_dishes_poisson_rate_posteriors[obs_idx, :]), one_tensor)

    bayesian_recursion_results = dict(
        table_assignment_priors=dish_eating_priors.numpy(),
        table_assignment_posteriors=dish_eating_posteriors.numpy(),
        table_assignment_posteriors_running_sum=dish_eating_posteriors_running_sum.numpy(),
        num_table_posteriors=num_dishes_poisson_rate_posteriors.numpy(),
        parameters={k: v.detach().numpy() for k, v in likelihood_params.items()},
    )

    return bayesian_recursion_results


def likelihood_linear_gaussian(torch_observation,
                               obs_idx,
                               likelihood_params):
    # TODO: figure out how to do gradient descent using the post-grad step means
    # covariances = torch.stack([
    #     torch.matmul(stddev, stddev.T) for stddev in likelihood_params['stddevs']])
    #
    obs_dim = torch_observation.shape[0]
    covariances = torch.stack([torch.matmul(torch.eye(obs_dim), torch.eye(obs_dim).T)
                               for stddev in likelihood_params['stddevs']]).double()

    mv_normal = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=likelihood_params['means'][:obs_idx + 1],
        covariance_matrix=covariances[:obs_idx + 1],
        # scale_tril=likelihood_params['stddevs'][:obs_idx + 1],
    )
    log_likelihoods_per_latent = mv_normal.log_prob(value=torch_observation)
    likelihoods_per_latent = torch.exp(log_likelihoods_per_latent)
    return likelihoods_per_latent, log_likelihoods_per_latent


def likelihood_multivariate_normal(torch_observation,
                                   obs_idx,
                                   likelihood_params):
    # TODO: figure out how to do gradient descent using the post-grad step means
    # covariances = torch.stack([
    #     torch.matmul(stddev, stddev.T) for stddev in likelihood_params['stddevs']])
    #
    obs_dim = torch_observation.shape[0]
    covariances = torch.stack([torch.matmul(torch.eye(obs_dim), torch.eye(obs_dim).T)
                               for stddev in likelihood_params['stddevs']]).double()

    mv_normal = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=likelihood_params['means'][:obs_idx + 1],
        covariance_matrix=covariances[:obs_idx + 1],
        # scale_tril=likelihood_params['stddevs'][:obs_idx + 1],
    )
    log_likelihoods_per_latent = mv_normal.log_prob(value=torch_observation)
    likelihoods_per_latent = torch.exp(log_likelihoods_per_latent)
    return likelihoods_per_latent, log_likelihoods_per_latent


def run_inference_alg(inference_alg_str: str,
                      observations: np.ndarray,
                      inference_params: dict,
                      likelihood_model: str,
                      learning_rate: float = 1e0,
                      likelihood_known: bool = True,
                      likelihood_params: dict = None):
    if likelihood_known:
        assert likelihood_params is not None

    # allow algorithm-specific arguments to inference alg function
    inference_alg_kwargs = dict()

    # select inference alg and add kwargs as necessary
    if inference_alg_str == 'R-IBP':
        inference_alg_fn = recursive_ibp
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
        inference_params=inference_params,
        likelihood_model=likelihood_model,
        learning_rate=learning_rate,
        likelihood_known=likelihood_known,
        likelihood_params=likelihood_params,
        **inference_alg_kwargs)

    return inference_alg_results
