from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
from typing import Dict

# common plotting functions
import utils.plot


def plot_analyze_all_algorithms_results(inf_algorithms_results_df: pd.DataFrame,
                                        plot_dir: str):
    plot_analyze_all_negative_posterior_predictive_vs_runtime(
        inf_algorithms_results_df=inf_algorithms_results_df,
        plot_dir=plot_dir)


def plot_analyze_all_negative_posterior_predictive_vs_runtime(inf_algorithms_results_df: pd.DataFrame,
                                                              plot_dir: str):
    # In alpha (X) vs beta (Y) space, plot runtime (grouping by algorithm,
    for sampling_scheme, results_by_sampling_df in inf_algorithms_results_df.groupby('sampling'):
        sampling_results_dir_path = os.path.join(plot_dir, sampling_scheme)

        sns.relplot(x='runtime',
                    y='negative_log_posterior_predictive',
                    hue='inference_alg',
                    data=results_by_sampling_df)
        plt.xlabel('Runtime')
        plt.ylabel('Negative Log Posterior Predictive')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(sampling_results_dir_path,
                                 'negative_posterior_predictive_vs_runtime.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
    print(10)


def plot_run_one_gaussian_features_by_num_obs(observations: np.ndarray,
                                              gaussian_features: np.ndarray,
                                              dish_eating_posteriors: np.ndarray,
                                              plot_dir: str,
                                              max_obs_idx: int = 20,
                                              max_num_features: int = 15):
    """
    Plot Gaussian 2-D features in grid, where each subplot in the grid
    is a different observation. Only plot the first max_obs_idx.

    Alpha is proportional to probability.
    """
    num_cols = 5
    num_rows = int(max_obs_idx / num_cols)

    # Gaussian features either have shape (num obs, max num features, ...) if the
    # inference algorithm is online, or shape (max num features, ...) if offline
    gaussian_features_have_obs_dim = True if len(gaussian_features.shape) == 3 else False
    if gaussian_features_have_obs_dim:
        actual_num_features = gaussian_features.shape[1]
    else:
        actual_num_features = gaussian_features.shape[0]
    max_num_features = min(max_num_features, actual_num_features)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(3 * num_cols, 3 * num_rows),
        sharex=True,
        sharey=True)

    for obs_idx in range(max_obs_idx):
        row_idx, col_idx = int(obs_idx / num_cols), obs_idx % num_cols
        ax = axes[row_idx, col_idx]
        for feature_idx in range(max_num_features):
            if gaussian_features_have_obs_dim:
                ax.plot([0, gaussian_features[obs_idx, feature_idx, 0]],
                        [0, gaussian_features[obs_idx, feature_idx, 1]],
                        label=f'{feature_idx}',
                        # alpha=dish_eating_posteriors[obs_idx, feature_idx],
                        )
            else:
                ax.plot([0, gaussian_features[feature_idx, 0]],
                        [0, gaussian_features[feature_idx, 1]],
                        label=f'{feature_idx}',
                        # alpha=dish_eating_posteriors[obs_idx, feature_idx],
                        )
        ax.scatter(observations[:obs_idx + 1, 0],
                   observations[:obs_idx + 1, 1],
                   s=3,
                   color='k',
                   label='Observations')
        ax.set_title(f'Obs {obs_idx + 1}')
        if row_idx == (num_rows - 1):
            ax.set_xlabel(r'$o_{1}$')
        if col_idx == 0:
            ax.set_ylabel(r'$o_{2}$')
    # plt.legend()
    plt.savefig(os.path.join(plot_dir, 'gaussian_features_by_num_obs.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_run_one_distance_btwn_true_features_and_inferred_feature_means(true_features: np.ndarray,
                                                                        inferred_feature_means: np.ndarray,
                                                                        plot_dir: str,
                                                                        metric_str: str = 'euclidean'):
    """
    Plo

    :param true_features: shape (num true features, feature dimension)
    :param inferred_feature_means: shape (num inferred features, feature dimension)
    :param plot_dir:
    :param metric_str:
    :return:
    """
    distances = cdist(true_features,
                      inferred_feature_means,
                      metric=metric_str)

    non_nan_columns = np.sum(np.isnan(distances), axis=0) == 0
    distances = distances[:, non_nan_columns]

    if distances.shape[1] > 0:
        ax = sns.heatmap(distances,
                         mask=np.isnan(distances),
                         # annot=True,
                         )
        ax.invert_yaxis()
        plt.ylabel('True Feature Index')
        plt.xlabel('Inferred Feature Index')
        plt.title(f'Distance: {metric_str}')
        plt.savefig(os.path.join(plot_dir, f'true_features_vs_inferred_feature_means_{metric_str}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_run_one_inference_results(sampled_linear_gaussian_data: dict,
                                   inference_alg_results: dict,
                                   inference_alg_str: str,
                                   inference_alg_params: Dict[str, float],
                                   log_posterior_predictive_dict: Dict[str, float],
                                   plot_dir):
    utils.plot.plot_run_one_num_features_by_num_obs(
        num_dishes_poisson_rate_priors=inference_alg_results['num_dishes_poisson_rate_priors'],
        num_dishes_poisson_rate_posteriors=inference_alg_results['num_dishes_poisson_rate_posteriors'],
        indicators=sampled_linear_gaussian_data['train_sampled_indicators'],
        plot_dir=plot_dir)

    utils.plot.plot_run_one_indicators_by_num_obs(
        dish_eating_priors=inference_alg_results['dish_eating_priors'],
        dish_eating_posteriors=inference_alg_results['dish_eating_posteriors'],
        indicators=sampled_linear_gaussian_data['train_sampled_indicators'],
        plot_dir=plot_dir)

    plot_run_one_gaussian_features_by_num_obs(
        observations=sampled_linear_gaussian_data['train_observations'],
        gaussian_features=inference_alg_results['inference_alg'].features_by_obs(),
        dish_eating_posteriors=inference_alg_results['dish_eating_posteriors'],
        plot_dir=plot_dir)

    plot_run_one_distance_btwn_true_features_and_inferred_feature_means(
        true_features=sampled_linear_gaussian_data['gaussian_params']['means'],
        inferred_feature_means=inference_alg_results['inference_alg'].features_after_last_obs(),
        metric_str='cosine',
        plot_dir=plot_dir)

    plot_run_one_true_features_vs_inferred_feature_means(
        observations=sampled_linear_gaussian_data['train_observations'],
        true_features=sampled_linear_gaussian_data['gaussian_params']['means'],
        inferred_feature_means=inference_alg_results['inference_alg'].features_after_last_obs(),
        plot_dir=plot_dir)


def plot_run_one_sample_from_linear_gaussian(features: np.ndarray,
                                             observations: np.ndarray,
                                             plot_dir: str):
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(6, 6))

    sns.scatterplot(observations[:, 0],
                    observations[:, 1],
                    palette='Set1',
                    ax=ax,
                    color='k',
                    legend=False, )

    for feature_idx, feature in enumerate(features):
        ax.plot([0, feature[0]],
                [0, feature[1]],
                alpha=0.5,
                label=f'Feature {feature_idx + 1}')

    ax.set_title('Ground Truth Data')
    ax.legend()
    plt.savefig(os.path.join(plot_dir, 'data.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_run_one_true_features_vs_inferred_feature_means(observations: np.ndarray,
                                                         true_features: np.ndarray,
                                                         inferred_feature_means: np.ndarray,
                                                         plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2)

    ax = axes[0]
    sns.scatterplot(observations[:, 0],
                    observations[:, 1],
                    palette='Set1',
                    ax=ax,
                    color='k',
                    legend=False)
    for feature_idx, feature in enumerate(true_features):
        ax.plot([0, feature[0]],
                [0, feature[1]],
                alpha=0.5,
                label=f'Feature {feature_idx + 1}')
    ax.set_title('True Features')

    ax = axes[1]
    sns.scatterplot(observations[:, 0],
                    observations[:, 1],
                    palette='Set1',
                    ax=ax,
                    color='k',
                    legend=False)
    for feature_idx, feature in enumerate(inferred_feature_means):
        ax.plot([0, feature[0]],
                [0, feature[1]],
                alpha=0.5,
                label=f'Feature {feature_idx + 1}')
    ax.set_title('Inferred Features')

    plt.savefig(os.path.join(plot_dir, 'true_features_vs_inferred_feature_means.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
