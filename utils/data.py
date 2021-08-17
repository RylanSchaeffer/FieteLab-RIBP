import numpy as np
import scipy.stats
from typing import Dict


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
        dtype=np.int8)
    cum_sampled_dishes_by_customer_idx = np.zeros(
        shape=(num_mc_sample, num_customer, max_dishes),
        dtype=np.int8)
    num_dishes_by_customer_idx = np.zeros(
        shape=(num_mc_sample, num_customer, max_dishes),
        dtype=np.int)

    for smpl_idx in range(num_mc_sample):
        current_num_dishes = 0
        for cstmr_idx in range(1, num_customer + 1):

            # sample existing dishes
            prob_new_customer_sampling_dish = cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx-2, :] /\
                                              (beta + cstmr_idx - 1)
            existing_dishes_for_new_customer = np.random.binomial(
                n=1,
                p=prob_new_customer_sampling_dish[np.newaxis, :])[0]
            sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx-1, :] = existing_dishes_for_new_customer # .astype(np.int)

            # sample number of new dishes for new customer
            # subtract 1 from to cstmr_idx because of 1-based iterating
            num_new_dishes = np.random.poisson(alpha * beta / (beta + cstmr_idx - 1))
            start_dish_idx = current_num_dishes
            end_dish_idx = current_num_dishes + num_new_dishes
            sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx-1, start_dish_idx:end_dish_idx] = 1

            # increment current num dishes
            current_num_dishes += num_new_dishes
            num_dishes_by_customer_idx[smpl_idx, cstmr_idx-1, current_num_dishes] = 1

            # increment running sums
            cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx-1, :] = np.add(
                cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx-2, :],
                sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx-1, :])

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


def sample_sequence_from_binary_linear_gaussian(class_sampling: str = 'Uniform',
                                                seq_len: int = 100,
                                                sigma_x_squared: float = 0.45,
                                                sigma_A: float = 10.94,
                                                obs_dim: int = 2,
                                                alpha: float = None,
                                                num_features: int = None):
    """
    IBP sampling.
    Draw sample from Binary Linear-Gaussian model, using either uniform sampling or

    Exactly one of alpha and num_latent_features must be specified.


    :param alpha:
    :param seq_len: desired sequence length
    :param sigma_x_squared:
    :param sigma_A:
    :param obs_dim:
    :param num_features:
    :return:
        assigned_table_seq: NumPy array with shape (seq_len,) of (integer) sampled classes
        linear_gaussian_samples_seq: NumPy array with shape (seq_len, obs_dim) of
                                binary linear-Gaussian samples
    """

    assert (alpha is None and num_features is not None) or \
           (alpha is not None and num_features is None)

    # sample binary Z (seq len \times K)
    if class_sampling == 'Uniform':
        assert num_features is not None
        binary_latent_features = np.random.randint(
            low=0,
            high=2,
            size=(seq_len, num_features))  # high is exclusive
        binary_latent_features = convert_binary_latent_features_to_left_order_form(
            binary_latent_features)
    elif class_sampling == 'IBP':
        assert alpha is not None
        binary_latent_features = sample_ibp(T=seq_len, alpha=alpha)
        # should be unnecessary, but just to check
        binary_latent_features = convert_binary_latent_features_to_left_order_form(
            binary_latent_features)
        num_features = np.sum(np.sum(binary_latent_features, axis=0) != 0)
    else:
        raise ValueError(f'Impermissible class sampling: {class_sampling}')

    # check all binary latent features are 0 or 1
    assert np.all(np.logical_or(binary_latent_features == 0, binary_latent_features == 1))

    # sample A (K \times D) from matrix normal
    feature_means = scipy.stats.matrix_normal.rvs(
        mean=np.zeros(shape=(num_features, obs_dim)),
        rowcov=sigma_A * np.eye(num_features),
        size=1)

    obs_means = np.matmul(binary_latent_features, feature_means)
    obs_cov = sigma_x_squared * np.eye(obs_dim)
    observations_seq = np.array([
        np.random.multivariate_normal(
            mean=obs_means[obs_idx],
            cov=obs_cov)
        for obs_idx in range(seq_len)])

    result = dict(
        observations_seq=observations_seq,
        binary_latent_features=binary_latent_features,
        feature_means=feature_means,
    )

    return result


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

    We make one modification. Paisely and Carin sample pi_k from
    Beta(beta_a/num_features, beta_b*(num_features-1)/num_features), which
    is an alternative parameterization of the IBP that does not agree with the
    parameterization used by Griffiths and Ghahramani (2011). Theirs samples pi_i from
    Beta(beta_a*beta_b/num_features, beta_b*(num_features-1)/num_features), which
    we will use here. Either is fine. You just need to be careful about which is used
    because the parameterization dictates the expected number of dishes per customer
    and the expected number of total dishes.

    Technically, we should take limit of num_features->infty. Instead we set
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

    observations_seq = np.matmul(np.multiply(indicators, weights), features) + noise

    assert observations_seq.shape == (seq_len, obs_dim)

    results = dict(
        observations_seq=observations_seq,
        indicators=indicators,
        features=features,
        feature_covariance=feature_covariance,
        weights=weights,
        weight_covariance=weight_covariance,
        noise=noise,
        obs_covariance=obs_covariance,
    )

    # import matplotlib.pyplot as plt
    # plt.scatter(observations_seq[:, 0],
    #             observations_seq[:, 1],
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
