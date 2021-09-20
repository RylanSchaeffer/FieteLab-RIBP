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
num_pcs = [100, 500, 1000, 1500]
dataset_sizes = [100, 250, 1000, 2500]
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

    frac_var_explained_by_num_pcs_and_dataset_size = []
    for num_pc in num_pcs:
        for dataset_size in dataset_sizes:
            rand_subset_indices = np.random.choice(
                possible_indices,
                size=dataset_size,
                replace=False)
            pca = PCA(n_components=min(num_pc, len(rand_subset_indices)))
            pca.fit_transform(features[rand_subset_indices])
            frac_var_explained = np.sum(pca.explained_variance_ratio_[:num_pc])
            frac_var_explained_by_num_pcs_and_dataset_size.append(
                (num_pc, dataset_size, frac_var_explained))
    df = pd.DataFrame(
        frac_var_explained_by_num_pcs_and_dataset_size,
        columns=['num_pc', 'dataset_size', 'frac_var_explained'])
    sns.lineplot(data=df,
                 x='dataset_size',
                 y='frac_var_explained',
                 hue='num_pc',
                 legend='full',  # necessary to force seaborn to not try binning based on hue
                 )
    plt.title(f'{dataset}')
    plt.xlabel('Dataset Size')
    plt.ylabel(f'Fraction of Variance Explained')
    plt.savefig(os.path.join(plot_dir,
                             f'{dataset.lower()}_frac_var_explained_by_fixed_num_pcs.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

