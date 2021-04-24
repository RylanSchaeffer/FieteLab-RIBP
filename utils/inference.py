import numpy as np
import torch

from utils.helpers import assert_torch_no_nan_no_inf, torch_logits_to_probs, torch_probs_to_logits

torch.set_default_tensor_type('torch.DoubleTensor')


def recursive_ibp(observations,
                  concentration_param: float,
                  likelihood_model: str,
                  learning_rate,
                  num_em_steps: int = 3):

    assert concentration_param > 0
    assert likelihood_model in {'multivariate_normal', 'dirichlet_multinomial',
                                'bernoulli', 'continuous_bernoulli'}
    num_obs, obs_dim = observations.shape

    # The recursion does not require recording the full history of priors/posteriors
    # but we record the full history for subsequent analysis
    max_num_latents = num_obs
    table_assignment_priors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)
    table_assignment_priors[0, 0] = 1.

    table_assignment_posteriors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    table_assignment_posteriors_running_sum = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    num_table_posteriors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    if likelihood_model == 'continuous_bernoulli':
        # need to use logits, otherwise gradient descent will carry parameters outside
        # valid interval
        cluster_parameters = dict(
            logits=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True)
        )
        create_new_cluster_params_fn = create_new_cluster_params_continuous_bernoulli
        likelihood_fn = likelihood_continuous_bernoulli

        # make sure no observation is 0 or 1 by adding epsilon
        epsilon = 1e-2
        observations[observations == 1.] -= epsilon
        observations[observations == 0.] += epsilon
    elif likelihood_model == 'dirichlet_multinomial':
        cluster_parameters = dict(
            topics_concentrations=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
        )
        create_new_cluster_params_fn = create_new_cluster_params_dirichlet_multinomial
        likelihood_fn = likelihood_dirichlet_multinomial
    elif likelihood_model == 'multivariate_normal':
        cluster_parameters = dict(
            means=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
            stddevs=torch.full(
                size=(max_num_latents, obs_dim, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
        )
        create_new_cluster_params_fn = create_new_cluster_params_multivariate_normal
        likelihood_fn = likelihood_multivariate_normal
    else:
        raise NotImplementedError

    # optimizer = torch.optim.SGD(params=[logits], lr=1.)
    optimizer = torch.optim.SGD(params=cluster_parameters.values(), lr=1.)

    # needed later for error checking
    one_tensor = torch.Tensor([1.]).double()

    torch_observations = torch.from_numpy(observations)
    for obs_idx, torch_observation in enumerate(torch_observations):

        # create new params for possible cluster, centered at that point
        create_new_cluster_params_fn(
            torch_observation=torch_observation,
            obs_idx=obs_idx,
            cluster_parameters=cluster_parameters)

        if obs_idx == 0:
            # first customer has to go at first table
            table_assignment_priors[obs_idx, 0] = 1.
            table_assignment_posteriors[obs_idx, 0] = 1.
            num_table_posteriors[obs_idx, 0] = 1.

            # update running sum of posteriors
            table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                table_assignment_posteriors_running_sum[obs_idx - 1, :],
                table_assignment_posteriors[obs_idx, :])
            assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
                                  torch.Tensor([obs_idx + 1]).double())
        else:
            # construct prior
            table_assignment_prior = torch.clone(
                table_assignment_posteriors_running_sum[obs_idx - 1, :obs_idx + 1])
            # we don't subtract 1 because Python uses 0-based indexing
            assert torch.allclose(torch.sum(table_assignment_prior), torch.Tensor([obs_idx]).double())
            # add new table probability
            table_assignment_prior[1:] += concentration_param * torch.clone(
                num_table_posteriors[obs_idx - 1, :obs_idx])
            # renormalize
            table_assignment_prior /= (concentration_param + obs_idx)
            assert torch.allclose(torch.sum(table_assignment_prior), one_tensor)

            # record latent prior
            table_assignment_priors[obs_idx, :len(table_assignment_prior)] = table_assignment_prior

            for em_idx in range(num_em_steps):

                optimizer.zero_grad()

                # E step: infer posteriors using parameters
                likelihoods_per_latent, log_likelihoods_per_latent = likelihood_fn(
                    torch_observation=torch_observation,
                    obs_idx=obs_idx,
                    cluster_parameters=cluster_parameters)
                assert torch.all(~torch.isnan(likelihoods_per_latent[:obs_idx + 1]))
                assert torch.all(~torch.isnan(log_likelihoods_per_latent[:obs_idx + 1]))

                unnormalized_table_assignment_posterior = torch.multiply(
                    likelihoods_per_latent.detach(),
                    table_assignment_prior)
                table_assignment_posterior = unnormalized_table_assignment_posterior / torch.sum(
                    unnormalized_table_assignment_posterior)
                assert torch.allclose(torch.sum(table_assignment_posterior), one_tensor)

                # record latent posterior
                table_assignment_posteriors[obs_idx, :len(table_assignment_posterior)] = table_assignment_posterior

                # update running sum of posteriors
                table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                    table_assignment_posteriors_running_sum[obs_idx - 1, :],
                    table_assignment_posteriors[obs_idx, :])
                assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
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
                    table_assignment_posteriors[obs_idx, :],
                    table_assignment_posteriors_running_sum[obs_idx, :]) / num_em_steps
                scaled_learning_rate[torch.isnan(scaled_learning_rate)] = 0.
                scaled_learning_rate[torch.isinf(scaled_learning_rate)] = 0.

                # don't update the newest cluster
                scaled_learning_rate[obs_idx] = 0.

                for param_descr, param_tensor in cluster_parameters.items():
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
            # for k1, p_z_t_equals_k1 in enumerate(table_assignment_posteriors[obs_idx, :obs_idx + 1]):
            #     for k2, p_prev_num_tables_equals_k2 in enumerate(num_table_posteriors[obs_idx - 1, :obs_idx + 1]):
            #         # advance number of tables by 1 if customer seating > number of current tables
            #         if k1 > k2 + 1:
            #             num_table_posteriors.data[obs_idx, k2 + 1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
            #         # customer allocated to previous table
            #         elif k1 <= k2:
            #             num_table_posteriors.data[obs_idx, k2] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
            #         # create new table
            #         elif k1 == k2 + 1:
            #             num_table_posteriors.data[obs_idx, k1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
            #         else:
            #             raise ValueError
            # assert torch.allclose(torch.sum(num_table_posteriors[obs_idx, :]), one_tensor)

            # new approach with time complexity O(t)
            # update posterior over number of tables using posterior over customer seat
            cum_table_assignment_posterior = torch.cumsum(
                table_assignment_posteriors[obs_idx, :obs_idx + 1],
                dim=0)
            one_minus_cum_table_assignment_posterior = 1. - cum_table_assignment_posterior
            prev_table_posterior = num_table_posteriors[obs_idx - 1, :obs_idx]
            num_table_posteriors[obs_idx, :obs_idx] += torch.multiply(
                cum_table_assignment_posterior[:-1],
                prev_table_posterior)
            num_table_posteriors[obs_idx, 1:obs_idx + 1] += torch.multiply(
                one_minus_cum_table_assignment_posterior[:-1],
                prev_table_posterior)
            assert torch.allclose(torch.sum(num_table_posteriors[obs_idx, :]), one_tensor)

    bayesian_recursion_results = dict(
        table_assignment_priors=table_assignment_priors.numpy(),
        table_assignment_posteriors=table_assignment_posteriors.numpy(),
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum.numpy(),
        num_table_posteriors=num_table_posteriors.numpy(),
        parameters={k: v.detach().numpy() for k, v in cluster_parameters.items()},
    )

    return bayesian_recursion_results



def run_inference_alg(inference_alg_str,
                      observations,
                      concentration_param,
                      likelihood_model,
                      learning_rate):
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
    #             dirichlet_concentration_param=10.)  # same as R-CRP
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
    #             dirichlet_concentration_param=10.)  # same as R-CRP
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
        concentration_param=concentration_param,
        likelihood_model=likelihood_model,
        learning_rate=learning_rate,
        **inference_alg_kwargs)

    return inference_alg_results