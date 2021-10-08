import joblib
import numpy as np
import os
import scipy.stats
import torch
import torchvision
from typing import Dict, Union


def convert_binary_latent_features_to_left_order_form(
        indicators: np.ndarray) -> np.ndarray:
    """
    Reorder to "Left Ordered Form" i.e. permute columns such that column
    as binary integers are decreasing

    :param indicators: shape (num customers, num dishes) of binary values
    """
    #
    # def left_order_form_indices_recursion(indicators_matrix, indices, row_idx):
    #     # https://stackoverflow.com/a/67699595/4570472
    #     if indices.size <= 1 or row_idx >= indicators_matrix.shape[0]:
    #         return indices
    #     left_indices = indices[np.where(indicators_matrix[row_idx, indices] == 1)]
    #     right_indices = indices[np.where(indicators_matrix[row_idx, indices] == 0)]
    #     return np.concatenate(
    #         (left_order_form_indices_recursion(indicators_matrix, indices=left_indices, row_idx=row_idx + 1),
    #          left_order_form_indices_recursion(indicators_matrix, indices=right_indices, row_idx=row_idx + 1)))

    # sort columns via recursion
    # reordered_indices = left_order_form_indices_recursion(
    #     indicators_matrix=indicators,
    #     row_idx=0,
    #     indices=np.arange(indicators.shape[1]))
    # left_ordered_indicators = indicators[:, reordered_indices]

    # sort columns via lexicographic sorting
    left_ordered_indicators_2 = indicators[:, np.lexsort(-indicators[::-1])]

    # check equality of both approaches
    # assert np.all(left_ordered_indicators == left_ordered_indicators_2)

    return left_ordered_indicators_2


def generate_gaussian_params_from_gaussian_prior(num_gaussians: int = 3,
                                                     gaussian_dim: int = 2,
                                                     gaussian_mean_prior_cov_scaling: float = 3.,
                                                     gaussian_cov_scaling: float = 0.3):
    # sample Gaussians' means from prior = N(0, rho * I)
    means = np.random.multivariate_normal(
        mean=np.zeros(gaussian_dim),
        cov=gaussian_mean_prior_cov_scaling * np.eye(gaussian_dim),
        size=num_gaussians)

    # all Gaussians have same covariance
    cov = gaussian_cov_scaling * np.eye(gaussian_dim)
    covs = np.repeat(cov[np.newaxis, :, :],
                     repeats=num_gaussians,
                     axis=0)

    mixture_of_gaussians = dict(means=means, covs=covs)

    return mixture_of_gaussians


def sample_ibp(num_mc_sample: int,
               num_customer: int,
               alpha: float,
               beta: float) -> Dict[str, np.ndarray]:
    assert alpha > 0.
    assert beta > 0.

    # preallocate results
    # use 10 * expected number of dishes as heuristic
    max_dishes = 10 * int(alpha * beta * np.sum(1 / (1 + np.arange(num_customer))))
    sampled_dishes_by_customer_idx = np.zeros(
        shape=(num_mc_sample, num_customer, max_dishes),
        dtype=np.int16)
    cum_sampled_dishes_by_customer_idx = np.zeros(
        shape=(num_mc_sample, num_customer, max_dishes),
        dtype=np.int16)
    num_dishes_by_customer_idx = np.zeros(
        shape=(num_mc_sample, num_customer, max_dishes),
        dtype=np.int16)

    for smpl_idx in range(num_mc_sample):
        current_num_dishes = 0
        for cstmr_idx in range(1, num_customer + 1):
            # sample existing dishes
            prob_new_customer_sampling_dish = cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 2, :] / \
                                              (beta + cstmr_idx - 1)
            existing_dishes_for_new_customer = np.random.binomial(
                n=1,
                p=prob_new_customer_sampling_dish[np.newaxis, :])[0]
            sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1,
            :] = existing_dishes_for_new_customer  # .astype(np.int)

            # sample number of new dishes for new customer
            # subtract 1 from to cstmr_idx because of 1-based iterating
            num_new_dishes = np.random.poisson(alpha * beta / (beta + cstmr_idx - 1))
            start_dish_idx = current_num_dishes
            end_dish_idx = current_num_dishes + num_new_dishes
            sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, start_dish_idx:end_dish_idx] = 1

            # increment current num dishes
            current_num_dishes += num_new_dishes
            num_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, current_num_dishes] = 1

            # increment running sums
            cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, :] = np.add(
                cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 2, :],
                sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, :])

    sample_ibp_results = {
        'cum_sampled_dishes_by_customer_idx': cum_sampled_dishes_by_customer_idx,
        'sampled_dishes_by_customer_idx': sampled_dishes_by_customer_idx,
        'num_dishes_by_customer_idx': num_dishes_by_customer_idx,
    }

    # import matplotlib.pyplot as plt
    # mc_avg = np.mean(sample_ibp_results['sampled_dishes_by_customer_idx'], axis=0)
    # plt.imshow(mc_avg)
    # plt.show()

    return sample_ibp_results


def sample_from_linear_gaussian(num_obs: int = 100,
                                indicator_sampling_str: str = 'categorical',
                                indicator_sampling_params: Dict[str, float] = None,
                                gaussian_prior_params: Dict[str, float] = None,
                                gaussian_likelihood_params: Dict[str, float] = None) -> Dict[str, np.ndarray]:
    """
    Draw sample from Binary Linear-Gaussian model.

    :return:
        sampled_indicators: NumPy array with shape (seq_len,) of (integer) sampled classes
        linear_gaussian_samples_seq: NumPy array with shape (seq_len, obs_dim) of
                                binary linear-Gaussian samples
    """

    if gaussian_prior_params is None:
        gaussian_prior_params = {}

    if gaussian_likelihood_params is None:
        gaussian_likelihood_params = {'sigma_x': 1e-10}

    # Otherwise, use categorical or IBP to sample number of features
    if indicator_sampling_str not in {'categorical', 'IBP', 'GriffithsGhahramani'}:
        raise ValueError(f'Impermissible class sampling value: {indicator_sampling_str}')

    # Unique case of generating Griffiths & Ghahramani data
    if indicator_sampling_str == 'GriffithsGhahramani':
        sampled_data_result = sample_from_griffiths_ghahramani_2005(
            num_obs=num_obs,
            gaussian_likelihood_params=gaussian_likelihood_params)
        sampled_data_result['indicator_sampling_str'] = indicator_sampling_str

        indicator_sampling_descr_str = '{}_probs=[{}]'.format(
            indicator_sampling_str,
            ','.join([str(i) for i in sampled_data_result['indicator_sampling_params']['probs']]))
        sampled_data_result['indicator_sampling_descr_str'] = indicator_sampling_descr_str
        return sampled_data_result

    if indicator_sampling_str is None:
        indicator_sampling_params = dict()

    if indicator_sampling_str == 'categorical':

        # if probabilities per cluster aren't specified, go with uniform probabilities
        if 'probs' not in indicator_sampling_params:
            indicator_sampling_params['probs'] = np.ones(5) / 5

        indicator_sampling_descr_str = '{}_probs={}'.format(
            indicator_sampling_str,
            list(indicator_sampling_params['probs']))
        indicator_sampling_descr_str = indicator_sampling_descr_str.replace(' ', '')

    elif indicator_sampling_str == 'IBP':
        if 'alpha' not in indicator_sampling_params:
            indicator_sampling_params['alpha'] = 3.98
        if 'beta' not in indicator_sampling_params:
            indicator_sampling_params['beta'] = 4.97
        indicator_sampling_descr_str = '{}_a={}_b={}'.format(
            indicator_sampling_str,
            indicator_sampling_params['alpha'],
            indicator_sampling_params['beta'])

    else:
        raise NotImplementedError

    if indicator_sampling_str == 'categorical':
        num_gaussians = indicator_sampling_params['probs'].shape[0]
        sampled_indicators = np.random.binomial(
            n=1,
            p=indicator_sampling_params['probs'][np.newaxis, :],
            size=(num_obs, num_gaussians))
    elif indicator_sampling_str == 'IBP':
        sampled_indicators = sample_ibp(
            num_mc_sample=1,
            num_customer=num_obs,
            alpha=indicator_sampling_params['alpha'],
            beta=indicator_sampling_params['beta'])['sampled_dishes_by_customer_idx'][0, :, :]
        num_gaussians = np.argwhere(np.sum(sampled_indicators, axis=0) == 0.)[0, 0]
        sampled_indicators = sampled_indicators[:, :num_gaussians]
    else:
        raise ValueError(f'Impermissible class sampling: {indicator_sampling_str}')

    gaussian_params = generate_gaussian_params_from_gaussian_prior(
        num_gaussians=num_gaussians,
        **gaussian_prior_params)

    features = gaussian_params['means']
    obs_dim = features.shape[1]
    obs_means = np.matmul(sampled_indicators, features)
    obs_cov = np.square(gaussian_likelihood_params['sigma_x']) * np.eye(obs_dim)
    observations = np.array([
        np.random.multivariate_normal(
            mean=obs_means[obs_idx],
            cov=obs_cov)
        for obs_idx in range(num_obs)])

    sampled_data_result = dict(
        gaussian_params=gaussian_params,
        sampled_indicators=sampled_indicators,
        observations=observations,
        features=features,
        indicator_sampling_str=indicator_sampling_str,
        indicator_sampling_params=indicator_sampling_params,
        indicator_sampling_descr_str=indicator_sampling_descr_str,
        gaussian_prior_params=gaussian_prior_params,
        gaussian_likelihood_params=gaussian_likelihood_params,
    )

    return sampled_data_result


def sample_from_griffiths_ghahramani_2005(num_obs: int = 100,
                                          indicator_sampling_params: Dict[str, Union[float, np.ndarray]] = None,
                                          gaussian_likelihood_params: Dict[str, float] = None):
    """
    Draw a sample from synthetic observations set used by Griffiths and Ghahramani 2005.

    Also used by Widjaja 2017 Stremaing VI for IBP.
    """

    if indicator_sampling_params is None:
        indicator_sampling_params = dict(probs=np.array([0.5, 0.5, 0.5, 0.5]))

    if gaussian_likelihood_params is None:
        gaussian_likelihood_params = dict(sigma_x=0.5)

    num_features = 4
    feature_dim = 36
    features = np.array([
        [
            [0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0]
        ]
    ], dtype='float64').reshape((num_features, feature_dim))

    # shape: 100 by number of features
    sampled_indicators = np.random.binomial(
        n=1,
        p=indicator_sampling_params['probs'][np.newaxis, :],
        size=(num_obs, num_features))
    observations = np.matmul(sampled_indicators, features)
    observations += scipy.stats.norm.rvs(loc=0.0, scale=gaussian_likelihood_params['sigma_x'], size=observations.shape)

    sampled_data_result = dict(
        sampled_indicators=sampled_indicators,
        observations=observations,
        features=features,
        indicator_sampling_params=indicator_sampling_params,
        gaussian_params={'means': features},
        original_features_shape=(num_features, 6, 6),  # sqrt(36)
    )

    return sampled_data_result


def sample_sequence_from_factor_analysis(seq_len: int,
                                         obs_dim: int = 25,
                                         max_num_features: int = 5000,
                                         beta_a: float = 1,  # copied from paper
                                         beta_b: float = 1,  # copied from paper
                                         weight_mean: float = 0.,
                                         weight_variance: float = 1.,
                                         obs_variance: float = 0.0675,  # copied from Paisely and Carin
                                         feature_covariance: np.ndarray = None):
    """Factor Analysis model from Paisley & Carin (2009) Equation 11.

    We make one modification. Paisley and Carin sample pi_k from
    Beta(beta_a/num_features, beta_b*(num_features-1)/num_features), which
    is an alternative parameterization of the IBP that does not agree with the
    parameterization used by Griffiths and Ghahramani (2011). Theirs samples pi_i from
    Beta(beta_a*beta_b/num_features, beta_b*(num_features-1)/num_features), which
    we will use here. Either is fine. You just need to be careful about which is used
    because the parameterization dictates the expected number of dishes per customer
    and the expected number of total dishes.

    Technically, we should take limit of num_features->infinity. Instead we set
    max_num_features very large, and keep only the nonzero ones.
    """

    if feature_covariance is None:
        feature_covariance = np.eye(obs_dim)

    pi = np.random.beta(a=beta_a * beta_b / max_num_features,
                        b=beta_b * (max_num_features - 1) / max_num_features,
                        size=max_num_features)
    # draw Z from Bernoulli i.e. Binomial with n=1
    indicators = np.random.binomial(n=1, p=pi, size=(seq_len, max_num_features))

    # convert to Left Ordered Form
    indicators = convert_binary_latent_features_to_left_order_form(
        indicators=indicators)

    # Uncomment to check correctness of indicators
    # num_dishes_per_customer = np.sum(indicators, axis=1)
    # average_dishes_per_customer = np.mean(num_dishes_per_customer)
    # average_dishes_per_customer_expected = beta_a
    non_empty_dishes = np.sum(indicators, axis=0)
    # total_dishes = np.sum(non_empty_dishes != 0)
    # total_dishes_expected = beta_a * beta_b * np.log(beta_b + seq_len)

    # only keep columns with non-empty dishes
    indicators = indicators[:, non_empty_dishes != 0]
    num_features = indicators.shape[1]

    weight_covariance = weight_variance * np.eye(num_features)
    weights = np.random.multivariate_normal(
        mean=weight_mean * np.ones(num_features),
        cov=weight_covariance,
        size=(seq_len,))

    features = np.random.multivariate_normal(
        mean=np.zeros(obs_dim),
        cov=feature_covariance,
        size=(num_features,))

    obs_covariance = obs_variance * np.eye(obs_dim)
    noise = np.random.multivariate_normal(
        mean=np.zeros(obs_dim),
        cov=obs_covariance,
        size=(seq_len,))

    assert indicators.shape == (seq_len, num_features)
    assert weights.shape == (seq_len, num_features)
    assert features.shape == (num_features, obs_dim)
    assert noise.shape == (seq_len, obs_dim)

    observations = np.matmul(np.multiply(indicators, weights), features) + noise

    assert observations.shape == (seq_len, obs_dim)

    results = dict(
        observations=observations,
        indicators=indicators,
        features=features,
        feature_covariance=feature_covariance,
        weights=weights,
        weight_covariance=weight_covariance,
        noise=noise,
        obs_covariance=obs_covariance,
    )

    # import matplotlib.pyplot as plt
    # plt.scatter(observations[:, 0],
    #             observations[:, 1],
    #             s=2)
    # # plot features
    # for k in range(num_features):
    #     plt.plot([0, pi[k]*features[k][0]],
    #              [0, pi[k]*features[k][1]],
    #              color='red',
    #              label='Scaled Features' if k == 0 else None)
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.legend()
    # plt.show()

    return results
