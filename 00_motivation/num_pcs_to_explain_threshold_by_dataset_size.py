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
threshold = 0.99
dataset_sizes = np.arange(100, 3501, 100)
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

    num_pcs_to_explain_var_by_dataset_size = []
    for dataset_size in dataset_sizes:
        rand_subset_indices = np.random.choice(
            possible_indices,
            size=dataset_size,
            replace=False)
        pca = PCA()
        pca.fit_transform(features[rand_subset_indices])
        cum_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        num_pc = np.argmax(cum_explained_variance_ratio > threshold)
        num_pcs_to_explain_var_by_dataset_size.append((dataset_size, num_pc))
        print(f'{dataset}: For dataset size {dataset_size}, {num_pc} PCs are'
              f' required to explain {threshold} of the variance')
    df = pd.DataFrame(
        num_pcs_to_explain_var_by_dataset_size,
        columns=['dataset_size', 'num_pcs'])
    sns.lineplot(data=df,
                 x='dataset_size',
                 y='num_pcs',
                 )
    plt.title(f'{dataset}')
    plt.xlabel('Dataset Size')
    plt.ylabel(f'Num PCs to Explain {100*threshold}% Variance')
    plt.savefig(os.path.join(plot_dir,
                             f'{dataset.lower()}_num_pcs_to_explain_{threshold}_by_dataset_size.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

