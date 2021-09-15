from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# common plotting functions
import utils.plot


def plot_gaussian_features_by_num_obs(observations: np.ndarray,
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
    actual_num_features = gaussian_features.shape[1]
    max_num_features = min(max_num_features, actual_num_features)

    # Gaussian features either have shape (num obs, max num features, ...) if the
    # inference algorithm is online, or shape (max num features, ...) if offline
    gaussian_features_have_obs_dim = True if len(gaussian_features) == 3 else False

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(3*num_cols, 3*num_rows),
        sharex=True,
        sharey=True)

    for obs_idx in range(max_obs_idx):
        row_idx, col_idx = int(obs_idx / num_cols), obs_idx % num_cols
        ax = axes[row_idx, col_idx]
        ax.scatter(observations[:obs_idx + 1, 0],
                   observations[:obs_idx + 1, 1],
                   s=1,
                   color='k',
                   label='Observations')
        ax.set_title(f'Obs {obs_idx+1}')
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


def plot_animation_gaussian_features_by_num_obs(observations: np.ndarray,
                                                gaussian_features: np.ndarray,
                                                plot_dir: str):
    """Create an animation of how Gaussian features evolve per observation"""
    print(10)

    fig, ax = plt.subplots()
    ud = UpdateDist(ax, prob=0.7)
    anim = FuncAnimation(fig, ud, frames=100, interval=100, blit=True)
    plt.show()


def plot_inference_results(sampled_linear_gaussian_data: dict,
                           inference_alg_results: dict,
                           inference_alg_str: str,
                           inference_alg_params: dict,
                           plot_dir):

    utils.plot.plot_posterior_num_dishes_by_num_obs(
        dish_eating_array=inference_alg_results['dish_eating_posteriors'],
        plot_dir=plot_dir)

    utils.plot.plot_poisson_rates_by_num_obs(
        num_dishes_poisson_rate_priors=inference_alg_results['num_dishes_poisson_rate_priors'],
        num_dishes_poisson_rate_posteriors=inference_alg_results['num_dishes_poisson_rate_posteriors'],
        indicators=sampled_linear_gaussian_data['sampled_indicators'],
        plot_dir=plot_dir)

    utils.plot.plot_indicators_by_num_obs(
        dish_eating_priors=inference_alg_results['dish_eating_priors'],
        dish_eating_posteriors=inference_alg_results['dish_eating_posteriors'],
        indicators=sampled_linear_gaussian_data['sampled_indicators'],
        plot_dir=plot_dir)

    plot_gaussian_features_by_num_obs(
        observations=sampled_linear_gaussian_data['observations'],
        gaussian_features=inference_alg_results['variable_parameters']['A']['mean'],
        dish_eating_posteriors=inference_alg_results['dish_eating_posteriors'],
        plot_dir=plot_dir)

    # plot_animation_gaussian_features_by_num_obs(
    #     observations=sampled_linear_gaussian_data['observations'],
    #     gaussian_features=inference_alg_results['variable_variational_params']['A']['mean'],
    #     plot_dir=plot_dir)


def plot_sample_from_linear_gaussian(features,
                                     observations,
                                     plot_dir):
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
