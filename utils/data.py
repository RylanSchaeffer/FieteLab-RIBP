import numpy as np
import scipy.stats


def convert_binary_latent_features_to_left_order_form(binary_latent_features):
    # reorder to "Left Ordered Form" i.e. permute columns such that column sums are decreasing
    seq_len = binary_latent_features.shape[0]
    powers_of_2 = np.power(2, np.arange(0, seq_len)[::-1], dtype=np.float64)
    column_sums = np.sum(np.multiply(binary_latent_features, powers_of_2[:, np.newaxis]), axis=0)
    column_ordering = np.argsort(column_sums)[::-1]
    return binary_latent_features[:, column_ordering]


def sample_sequence_from_ibp(T: int,
                             alpha: float):
    # shape: (number of customers, number of dishes)
    # heuristic: 3 * expected number
    max_dishes = int(3 * alpha * np.sum(1 / (1 + np.arange(T))))
    customers_dishes_draw = np.zeros(shape=(T, max_dishes), dtype=np.int)

    current_num_dishes = 0
    for t in range(T):
        # sample old dishes for new customer
        frac_prev_customers_sampling_dish = np.sum(customers_dishes_draw[:t, :], axis=0) / (t + 1)
        dishes_for_new_customer = np.random.binomial(n=1, p=frac_prev_customers_sampling_dish[np.newaxis, :])[0]
        customers_dishes_draw[t, :] = dishes_for_new_customer.astype(np.int)

        # sample number of new dishes for new customer
        # add +1 to t because of 0-based indexing
        num_new_dishes = np.random.poisson(alpha / (t + 1))
        customers_dishes_draw[t, current_num_dishes:current_num_dishes + num_new_dishes] = 1

        # increment current num dishes
        current_num_dishes += num_new_dishes

    return customers_dishes_draw


vectorized_sample_sequence_from_ibp = np.vectorize(sample_sequence_from_ibp,
                                                   otypes=[np.ndarray])


def sample_sequence_from_binary_linear_gaussian(class_sampling: str = 'Uniform',
                                                seq_len: int = 100,
                                                sigma_x_squared: float = 0.45,
                                                sigma_A: float = 10.94,
                                                obs_dim: int = 2,
                                                alpha: float = None,
                                                num_latent_features: int = None):
    """
    Draw sample from Binary Linear-Gaussian model, using either uniform sampling or
    IBP sampling.
    
    Exactly one of alpha and num_latent_features must be specified.


    :param alpha:
    :param seq_len: desired sequence length
    :param sigma_x_squared:
    :param sigma_A:
    :param obs_dim:
    :param num_latent_features:
    :return:
        assigned_table_seq: NumPy array with shape (seq_len,) of (integer) sampled classes
        linear_gaussian_samples_seq: NumPy array with shape (seq_len, obs_dim) of
                                binary linear-Gaussian samples
    """

    assert (alpha is None and num_latent_features is not None) or \
           (alpha is not None and num_latent_features is None)

    # sample binary Z (seq len \times K)
    if class_sampling == 'Uniform':
        assert num_latent_features is not None
        binary_latent_features = np.random.randint(
            low=0,
            high=2,
            size=(seq_len, num_latent_features))  # high is exclusive
        binary_latent_features = convert_binary_latent_features_to_left_order_form(
            binary_latent_features)
    elif class_sampling == 'IBP':
        assert alpha is not None
        binary_latent_features = sample_sequence_from_ibp(T=seq_len, alpha=alpha)
        # should be unnecessary, but just to check
        binary_latent_features = convert_binary_latent_features_to_left_order_form(
            binary_latent_features)
        num_latent_features = np.sum(np.sum(binary_latent_features, axis=0) != 0)
    else:
        raise ValueError(f'Impermissible class sampling: {class_sampling}')

    # check all binary latent features are 0 or 1
    assert np.all(np.logical_or(binary_latent_features == 0, binary_latent_features == 1))

    # sample A (K \times D) from matrix normal
    feature_means = scipy.stats.matrix_normal.rvs(
        mean=np.zeros(shape=(num_latent_features, obs_dim)),
        rowcov=sigma_A * np.eye(num_latent_features),
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
