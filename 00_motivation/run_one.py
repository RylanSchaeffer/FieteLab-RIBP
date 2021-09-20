"""
For MNIST and Omniglot, plot the cumulative fraction of variance explained (Y)
vs the principal component index (X) for varying dataset sizes (hue).

Example usage:

python3 00_motivation/run_one.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

import utils.data_real

plot_dir = '00_motivation'


datasets = ['Omniglot', 'MNIST']
for dataset in datasets:
    if dataset == 'MNIST':
        mnist_results = utils.data_real.load_mnist_dataset(
            feature_extractor_method=None)
        features = mnist_results['image_features']
        possible_indices = np.arange(features.shape[0])
    elif dataset == 'Omniglot':
        omniglot_results = utils.data_real.load_omniglot_dataset(
            feature_extractor_method=None,
            center_crop=False)
        features = omniglot_results['image_features']
        possible_indices = np.arange(features.shape[0])
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    subset_sizes = [100, 250, 1000, 2500]
    cum_explained_variance_ratio_by_subset_size = dict()
    for subset_size in subset_sizes:
        rand_subset_indices = np.random.choice(
            possible_indices,
            size=subset_size,
            replace=False)
        pca = PCA(n_components=100)
        pca.fit_transform(features[rand_subset_indices])
        cum_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        cum_explained_variance_ratio_by_subset_size[subset_size] = cum_explained_variance_ratio
    df = pd.DataFrame.from_dict(
        cum_explained_variance_ratio_by_subset_size)
    df['index'] = df.index + 1
    melted_df = df.melt(
            id_vars=['index'],  # columns to keep
            var_name='subset_size',  # new column name for previous columns' headers
            value_name='frac_explained_variance',  # new column name for values
        )

    sns.lineplot(data=melted_df,
                 x='index',
                 y='frac_explained_variance',
                 hue='subset_size',
                 legend='full',  # necessary to force seaborn to not try binning based on hue
                 )
    plt.title(f'{dataset}')
    plt.xlabel('PC Index')
    plt.ylabel('Fraction of Variance Explained')
    plt.savefig(os.path.join(plot_dir,
                             f'{dataset.lower()}_frac_var_explained_by_pc_index.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

