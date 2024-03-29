"""
For MNIST and Omniglot, plot the cumulative fraction of variance explained (Y)
vs the principal component index (X) for varying dataset sizes (hue).

Example usage:

python3 00_motivation/frac_var_explained_by_fixed_num_pcs.py
"""

import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA

import utils.data.real


plot_dir = '00_motivation/results'
os.makedirs(plot_dir, exist_ok=True)

datasets = ['Omniglot', 'MNIST']
num_pcs = [100, 250, 500, 1000]
dataset_sizes = np.arange(100, 3501, 100)
for dataset in datasets:
    if dataset == 'MNIST':
        mnist_results = utils.data.real.load_dataset_mnist(
            feature_extractor_method=None)
        features = mnist_results['image_features']
        possible_indices = np.arange(features.shape[0])
    elif dataset == 'Omniglot':
        omniglot_results = utils.data.real.load_dataset_omniglot(
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

            # Plot every time, just in case execution is interrupted
            df = pd.DataFrame(
                frac_var_explained_by_num_pcs_and_dataset_size,
                columns=['Num PCs', 'dataset_size', 'frac_var_explained'])
            df.to_csv(os.path.join(plot_dir, f'{dataset.lower()}_frac_var_explained_by_fixed_num_pcs.csv'),
                      index=False)
