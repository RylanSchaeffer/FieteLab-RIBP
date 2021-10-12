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

import utils.real

plot_dir = '00_motivation/results'
os.makedirs(plot_dir, exist_ok=True)

datasets = ['Omniglot', 'MNIST']
thresholds = [0.5, 0.75, 0.95, 0.99]
dataset_sizes = np.arange(100, 3501, 100)
for dataset in datasets:
    if dataset == 'MNIST':
        mnist_results = utils.real.load_mnist_dataset(
            feature_extractor_method=None)
        features = mnist_results['image_features']
        possible_indices = np.arange(features.shape[0])
    elif dataset == 'Omniglot':
        omniglot_results = utils.real.load_omniglot_dataset(
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
        for threshold in thresholds:
            num_pc = np.argmax(cum_explained_variance_ratio > threshold)
            num_pcs_to_explain_var_by_dataset_size.append((threshold, dataset_size, num_pc))
            print(f'{dataset}: For dataset size {dataset_size}, {num_pc} PCs are'
                  f' required to explain {threshold} of the variance')

        # Plot and save, in case it breaks
        df = pd.DataFrame(
            num_pcs_to_explain_var_by_dataset_size,
            columns=['threshold', 'dataset_size', 'num_pcs'])
        g = sns.lineplot(data=df,
                         x='dataset_size',
                         y='num_pcs',
                         hue='threshold',
                         legend='full',  # necessary to force seaborn to not try binning based on hue
                         )
        plt.title(f'{dataset}')
        plt.xlabel('Dataset Size')
        plt.ylabel(f'Num PCs to Explain % Variance')
        legend = g.legend()
        legend.texts[0].set_text("% Variance")
        plt.savefig(os.path.join(plot_dir,
                                 f'{dataset.lower()}_num_pcs_to_explain_by_dataset_size.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
