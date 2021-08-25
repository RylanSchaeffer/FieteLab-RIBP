from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# common plotting functions
import utils.plot


def plot_inference_results(sampled_linear_gaussian_data: dict,
                           inference_alg_results: dict,
                           inference_alg_str: str,
                           inference_alg_params: dict,
                           plot_dir):

    utils.plot.plot_num_dishes_by_num_obs(
        dish_eating_array=inference_alg_results['dish_eating_posteriors'],
        plot_dir=plot_dir)

    utils.plot.plot_poisson_rates_by_num_obs(
        num_dishes_poisson_rate_priors=inference_alg_results['num_dishes_poisson_rate_priors'],
        num_dishes_poisson_rate_posteriors=inference_alg_results['num_dishes_poisson_rate_posteriors'],
        indicators=sampled_linear_gaussian_data['sampled_indicators'],
        plot_dir=plot_dir)

    plt.title(f'Obs Idx: {obs_idx}, VI Idx: {vi_idx + 1}')
    plt.scatter(np.arange(len(dish_eating_prior)), dish_eating_prior, label='prior')
    plt.scatter(np.arange(len(dish_eating_prior)), Z_probs.detach().numpy(), label='posterior')
    plt.legend()
    plt.xlim(0, 15)
    plt.show()

    plt.title(f'Obs Idx: {obs_idx}, VI Idx: {vi_idx + 1}')
    plt.scatter(observations[:obs_idx, 0],
                observations[:obs_idx, 1],
                color='k',
                label='Observations')
    for feature_idx in range(15):
        plt.plot([0, variable_variational_params['A']['mean'][obs_idx, feature_idx, 0].data],
                 [0, variable_variational_params['A']['mean'][obs_idx, feature_idx, 1].data],
                 label=f'{feature_idx}')
    # plt.legend()
    plt.show()


def plot_sample_from_linear_gaussian(features,
                                     observations_seq,
                                     plot_dir):

    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(6, 6))

    sns.scatterplot(observations_seq[:, 0],
                    observations_seq[:, 1],
                    palette='Set1',
                    ax=ax,
                    color='k',
                    legend=False,)

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
    plt.show()
    plt.close()
