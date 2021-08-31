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


def generate_gaussian_parameters_from_gaussian_prior(num_gaussians: int = 3,
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


def load_moseq_dataset(data_dir: str = 'data'):
    """
    Load MoSeq data from https://github.com/dattalab/moseq-drugs.

    Note: You need to download the data from https://doi.org/10.5281/zenodo.3951698.
    Unfortunately, the data appears to have been pickled using Python2.7 and now
    NumPy and joblib can't read it. I used Python2.7 and Joblib to dump the data
    to disk using the following code:


    moseq_dir = os.path.join('data/moseq_drugs')
    file_names = [
        'dataset',
        'fingerprints',
        'syllablelabels'
    ]
    data = dict()
    for file_name in file_names:
        data[file_name] = np.load(os.path.join(moseq_dir, file_name + '.pkl'),
                                  allow_pickle=True)
    joblib.dump(value=data,
                filename=os.path.join(moseq_dir, 'moseq_drugs_data.joblib'))
    """

    moseq_dir = os.path.join(data_dir, 'moseq_drugs')
    data = joblib.load(filename=os.path.join(moseq_dir, 'moseq_drugs_data.joblib'))
    dataset = data['dataset']
    fingerprints = data['fingerprints']
    syllablelabels = data['syllablelabels']

    moseq_dataset_results = dict(
        dataset=dataset,
        fingerprints=fingerprints,
        syllablelabels=syllablelabels)

    return moseq_dataset_results


def load_newsgroup_dataset(data_dir: str = 'data',
                           num_data: int = None,
                           num_features: int = 500,
                           tf_or_tfidf_or_counts: str = 'tfidf'):
    assert tf_or_tfidf_or_counts in {'tf', 'tfidf', 'counts'}

    # categories = ['soc.religion.christian', 'comp.graphics', 'sci.med']
    categories = None  # set to None for all categories

    twenty_train = sklearn.datasets.fetch_20newsgroups(
        data_home=data_dir,
        subset='train',  # can switch to 'test'
        categories=categories,
        shuffle=True,
        random_state=0)

    class_names = np.array(twenty_train.target_names)
    true_cluster_labels = twenty_train.target
    true_cluster_label_strs = class_names[true_cluster_labels]
    observations = twenty_train.data

    if num_data is None:
        num_data = len(class_names)
    observations = observations[:num_data]
    true_cluster_labels = true_cluster_labels[:num_data]
    true_cluster_label_strs = true_cluster_label_strs[:num_data]

    if tf_or_tfidf_or_counts == 'tf':
        feature_extractor = sklearn.feature_extraction.text.TfidfVectorizer(
            max_features=num_features,  # Lin 2013 used 5000
            sublinear_tf=False,
            use_idf=False,
        )
        observations_transformed = feature_extractor.fit_transform(observations)

    elif tf_or_tfidf_or_counts == 'tfidf':
        # convert documents' word counts to tf-idf (Term Frequency times Inverse Document Frequency)
        # equivalent to CountVectorizer() + TfidfTransformer()
        # for more info, see
        # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
        # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
        feature_extractor = sklearn.feature_extraction.text.TfidfVectorizer(
            max_features=num_features,  # Lin 2013 used 5000
            sublinear_tf=False,
            use_idf=True,
        )
        observations_transformed = feature_extractor.fit_transform(observations)
    elif tf_or_tfidf_or_counts == 'counts':
        feature_extractor = sklearn.feature_extraction.text.CountVectorizer(
            max_features=num_features)
        observations_transformed = feature_extractor.fit_transform(observations)
    else:
        raise ValueError

    # quoting from Lin NeurIPS 2013:
    # We pruned the vocabulary to 5000 words by removing stop words and
    # those with low TF-IDF scores, and obtained 150 topics by running LDA [3]
    # on a subset of 20K documents. We held out 10K documents for testing and use the
    # remaining to train the DPMM. We compared SVA,SVA-PM, and TVF.

    # possible likelihoods for TF-IDF data
    # https://stats.stackexchange.com/questions/271923/how-to-use-tfidf-vectors-with-multinomial-naive-bayes
    # https://stackoverflow.com/questions/43237286/how-can-we-use-tfidf-vectors-with-multinomial-naive-bayes
    newsgroup_dataset_results = dict(
        observations_transformed=observations_transformed.toarray(),  # remove .toarray() to keep CSR matrix
        true_cluster_label_strs=true_cluster_label_strs,
        assigned_table_seq=true_cluster_labels,
        feature_extractor=feature_extractor,
        feature_names=feature_extractor.get_feature_names(),
    )

    return newsgroup_dataset_results


def load_omniglot_dataset(data_dir: str = 'data',
                          num_data: int = None,
                          center_crop: bool = True,
                          avg_pool: bool = False,
                          feature_extractor_method: str = 'pca'):
    """

    """

    assert feature_extractor_method in {'pca', 'cnn', 'vae', 'vae_old'}

    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    transforms = [torchvision.transforms.ToTensor()]
    if center_crop:
        transforms.append(torchvision.transforms.CenterCrop((80, 80)))
    if avg_pool:
        transforms.append(torchvision.transforms.Lambda(
            lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=9, stride=3)))

    omniglot_dataset = torchvision.datasets.Omniglot(
        root=data_dir,
        download=True,
        transform=torchvision.transforms.Compose(transforms))

    # truncate dataset for now
    # character_classes = [images_and_classes[1] for images_and_classes in
    #                      omniglot_dataset._flat_character_images]
    if num_data is None:
        num_data = len(omniglot_dataset._flat_character_images)
    omniglot_dataset._flat_character_images = omniglot_dataset._flat_character_images[:num_data]
    dataset_size = len(omniglot_dataset._flat_character_images)

    omniglot_dataloader = torch.utils.data.DataLoader(
        dataset=omniglot_dataset,
        batch_size=1,
        shuffle=False)

    images, labels = [], []
    for image, label in omniglot_dataloader:
        labels.append(label)
        images.append(image[0, 0, :, :])
        # uncomment to deterministically append the first image
        # images.append(omniglot_dataset[0][0][0, :, :])
    images = torch.stack(images).numpy()

    # these might be swapped but I think height = width for omniglot
    _, image_height, image_width = images.shape
    labels = np.array(labels)

    if feature_extractor_method == 'pca':
        from sklearn.decomposition import PCA
        reshaped_images = np.reshape(images, newshape=(dataset_size, image_height * image_width))
        pca = PCA(n_components=20)
        pca_latents = pca.fit_transform(reshaped_images)
        image_features = np.reshape(pca.inverse_transform(pca_latents),
                                    newshape=(dataset_size, image_height, image_width))
        feature_extractor = pca
    elif feature_extractor_method == 'cnn':
        # # for some reason, omniglot uses 1 for background and 0 for stroke
        # # whereas MNIST uses 0 for background and 1 for stroke
        # # for consistency, we'll invert omniglot
        # images = 1. - images
        #
        # from utils.omniglot_feature_extraction import cnn_load
        # lenet = cnn_load()
        #
        # from skimage.transform import resize
        # downsized_images = np.stack([resize(image, output_shape=(28, 28))
        #                              for image in images])
        #
        # # import matplotlib.pyplot as plt
        # # plt.imshow(downsized_images[0], cmap='gray')
        # # plt.title('Test Downsized Omniglot')
        # # plt.show()
        #
        # # add channel dimension for CNN
        # reshaped_images = np.expand_dims(downsized_images, axis=1)
        #
        # # make sure dropout is turned off
        # lenet.eval()
        # image_features = lenet(torch.from_numpy(reshaped_images)).detach().numpy()
        #
        # feature_extractor = lenet

        raise NotImplementedError
    elif feature_extractor_method == 'vae':
        vae_data = np.load('data/omniglot_vae/omniglot_data.npz')
        labels = vae_data['targets']
        indices_to_sort_labels = np.argsort(labels)
        # sort and truncate
        labels = labels[indices_to_sort_labels][:num_data]
        images = vae_data['images'][indices_to_sort_labels][:num_data, :, :]
        image_features = vae_data['latents'][indices_to_sort_labels][:num_data, :]
        feature_extractor = None
    else:
        raise ValueError(f'Impermissible feature method: {feature_extractor_method}')

    # # visualize images if curious
    # import matplotlib.pyplot as plt
    # for idx in range(10):
    #     plt.imshow(image_features[idx], cmap='gray')
    #     plt.show()

    omniglot_dataset_results = dict(
        images=images,
        assigned_table_seq=labels,
        feature_extractor_method=feature_extractor_method,
        feature_extractor=feature_extractor,
        image_features=image_features,
    )

    return omniglot_dataset_results


def load_reddit_dataset(num_data: int,
                        num_features: int,
                        tf_or_tfidf_or_counts='tfidf',
                        data_dir='data'):
    # TODO: rewrite this function to preprocess data similar to newsgroup
    os.makedirs(data_dir, exist_ok=True)

    # possible other alternative datasets:
    #   https://www.tensorflow.org/datasets/catalog/cnn_dailymail
    #   https://www.tensorflow.org/datasets/catalog/newsroom (also in sklearn)

    # useful overview: https://www.tensorflow.org/datasets/overview
    # take only subset of data for speed: https://www.tensorflow.org/datasets/splits
    # specific dataset: https://www.tensorflow.org/datasets/catalog/reddit
    reddit_dataset, reddit_dataset_info = tfds.load(
        'reddit',
        split='train',  # [:1%]',
        shuffle_files=False,
        download=True,
        with_info=True,
        data_dir=data_dir)
    assert isinstance(reddit_dataset, tf.data.Dataset)
    # reddit_dataframe = pd.DataFrame(reddit_dataset.take(10))
    reddit_dataframe = tfds.as_dataframe(
        ds=reddit_dataset.take(num_data),
        ds_info=reddit_dataset_info)
    reddit_dataframe = pd.DataFrame(reddit_dataframe)

    true_cluster_label_strs = reddit_dataframe['subreddit'].values
    true_cluster_labels = reddit_dataframe['subreddit'].astype('category').cat.codes.values

    documents_text = reddit_dataframe['normalizedBody'].values

    # convert documents' word counts to tf-idf (Term Frequency times Inverse Document Frequency)
    # equivalent to CountVectorizer() + TfidfTransformer()
    # for more info, see
    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        max_features=5000,
        sublinear_tf=False)
    observations_transformed = tfidf_vectorizer.fit_transform(documents_text)

    # quoting from Lin NeurIPS 2013:
    # We pruned the vocabulary to 5000 words by removing stop words and
    # those with low TF-IDF scores, and obtained 150 topics by running LDA [3]
    # on a subset of 20K documents. We held out 10K documents for testing and use the
    # remaining to train the DPMM. We compared SVA,SVA-PM, and TVF.

    # possible likelihoods for TF-IDF data
    # https://stats.stackexchange.com/questions/271923/how-to-use-tfidf-vectors-with-multinomial-naive-bayes
    # https://stackoverflow.com/questions/43237286/how-can-we-use-tfidf-vectors-with-multinomial-naive-bayes
    reddit_dataset_results = dict(
        observations_transformed=observations_transformed.toarray(),  # remove .toarray() to keep CSR matrix
        true_cluster_label_strs=true_cluster_label_strs,
        assigned_table_seq=true_cluster_labels,
        tfidf_vectorizer=tfidf_vectorizer,
        feature_names=tfidf_vectorizer.get_feature_names(),
    )

    return reddit_dataset_results


def load_yale_dataset(num_data: int,
                      data_dir='data'):
    npzfile = np.load(os.path.join('data', 'yale_faces', 'yale_faces.npz'))
    data = dict(npzfile)
    train_data = data['train_data']
    test_data = data['test_data']
    # authors suggest withholding pixels from testing set with 0.3% probability
    # these are those pixels
    test_mask = data['test_mask']

    # image size is 32x32. Reshape or no?

    yale_dataset_results = dict(
        train_data=train_data,
        test_data=test_data,
        test_mask=test_mask)

    return yale_dataset_results


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
                                indicator_sampling: str = 'categorical',
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

    if indicator_sampling not in {'categorical', 'IBP'}:
        raise ValueError(f'Impermissible class sampling value: {indicator_sampling}')

    if indicator_sampling is None:
        indicator_sampling_params = dict()

    if indicator_sampling == 'categorical':

        # if probabilities per cluster aren't specified, go with uniform probabilities
        if 'probs' not in indicator_sampling_params:
            indicator_sampling_params['probs'] = np.ones(5) / 5

        indicator_sampling_descr_str = '{}_probs={}'.format(
            indicator_sampling,
            list(indicator_sampling_params['probs']))
        indicator_sampling_descr_str = indicator_sampling_descr_str.replace(' ', '')

    elif indicator_sampling == 'IBP':
        if 'alpha' not in indicator_sampling_params:
            indicator_sampling_params['alpha'] = 3.98
        if 'beta' not in indicator_sampling_params:
            indicator_sampling_params['beta'] = 4.97
        indicator_sampling_descr_str = '{}_a={}_b={}'.format(
            indicator_sampling,
            indicator_sampling_params['alpha'],
            indicator_sampling_params['beta'])

    else:
        raise NotImplementedError

    if gaussian_prior_params is None:
        gaussian_prior_params = {}

    if gaussian_likelihood_params is None:
        gaussian_likelihood_params = {'sigma_x': 1e-10}

    if indicator_sampling == 'categorical':
        num_gaussians = indicator_sampling_params['probs'].shape[0]
        sampled_indicators = np.random.binomial(
            n=1,
            p=indicator_sampling_params['probs'][np.newaxis, :],
            size=(num_obs, num_gaussians))
    elif indicator_sampling == 'IBP':
        sampled_indicators = sample_ibp(
            num_mc_sample=1,
            num_customer=num_obs,
            alpha=indicator_sampling_params['alpha'],
            beta=indicator_sampling_params['beta'])['sampled_dishes_by_customer_idx'][0, :, :]
        num_gaussians = np.argwhere(np.sum(sampled_indicators, axis=0) == 0.)[0, 0]
        sampled_indicators = sampled_indicators[:, :num_gaussians]
    else:
        raise ValueError(f'Impermissible class sampling: {indicator_sampling}')

    gaussian_parameters = generate_gaussian_parameters_from_gaussian_prior(
        num_gaussians=num_gaussians,
        **gaussian_prior_params)

    features = gaussian_parameters['means']
    obs_dim = features.shape[1]
    obs_means = np.matmul(sampled_indicators, features)
    obs_cov = np.square(gaussian_likelihood_params['sigma_x']) * np.eye(obs_dim)
    observations_seq = np.array([
        np.random.multivariate_normal(
            mean=obs_means[obs_idx],
            cov=obs_cov)
        for obs_idx in range(num_obs)])

    result = dict(
        gaussian_parameters=gaussian_parameters,
        sampled_indicators=sampled_indicators,
        observations_seq=observations_seq,
        features=features,
        indicator_sampling=indicator_sampling,
        indicator_sampling_params=indicator_sampling_params,
        indicator_sampling_descr_str=indicator_sampling_descr_str,
        gaussian_prior_params=gaussian_prior_params,
        gaussian_likelihood_params=gaussian_likelihood_params,
    )

    return result


def sample_from_griffiths_ghahramani_2005(num_obs: int = 100,
                                          indicator_sampling_params: Dict[str, Union[float, np.ndarray]] = None,
                                          gaussian_likelihood_params: Dict[str, float] = None):
    """
    Draw a sample from synthetic observations_seq set used by Griffiths and Ghahramani 2005.

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
    observations_seq = np.matmul(sampled_indicators, features)
    observations_seq += scipy.stats.norm.rvs(loc=0.0, scale=gaussian_likelihood_params['sigma_x'], size=observations_seq.shape)

    result = dict(
        sampled_indicators=sampled_indicators,
        observations_seq=observations_seq,
        features=features,
        indicator_sampling_params=indicator_sampling_params,
        gaussian_likelihood_params=gaussian_likelihood_params,
        original_features_shape=(num_features, 6, 6),  # sqrt(36)
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
