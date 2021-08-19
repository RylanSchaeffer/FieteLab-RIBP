from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


# common plotting functions


def plot_inference_results(sampled_factor_analysis_results: dict,
                           inference_results: dict,
                           inference_alg_str: str,
                           inference_params: dict,
                           plot_dir,
                           num_tables_to_plot: int = 10):
    plot_inferred_dishes(
        indicators=sampled_factor_analysis_results['indicators'],
        dish_eating_priors=inference_results['dish_eating_priors'],
        dish_eating_posteriors=inference_results['dish_eating_posteriors'],
        plot_dir=plot_dir)

    plot_num_dishes_by_observation(
        indicators=sampled_factor_analysis_results['indicators'],
        num_dishes_poisson_rate_priors=inference_results['num_dishes_poisson_rate_priors'],
        num_dishes_poisson_rate_posteriors=inference_results['num_dishes_poisson_rate_posteriors'],
        plot_dir=plot_dir)


def plot_inferred_dishes(indicators,
                         dish_eating_priors,
                         dish_eating_posteriors,
                         plot_dir):
    fig, axes = plt.subplots(nrows=1,
                             ncols=5,
                             figsize=(12, 5),
                             gridspec_kw={'width_ratios': [1, 0.25, 1, 0.25, 1]})
    cutoff = 1e-2
    indicators = indicators.astype(float)
    indicators[indicators < cutoff] = np.nan
    axes[0].set_title(r'$z_{tk}$')
    sns.heatmap(indicators,
                ax=axes[0],
                mask=np.isnan(indicators),
                cmap='jet',
                vmin=cutoff,
                vmax=1.,
                norm=LogNorm())

    axes[1].axis('off')

    dish_eating_priors[dish_eating_priors < cutoff] = np.nan
    axes[2].set_title(r'$p(z_{tk}=1|o_{<t})$')
    sns.heatmap(dish_eating_priors,
                ax=axes[2],
                mask=np.isnan(dish_eating_priors),
                cmap='jet',
                vmin=cutoff,
                vmax=1.,
                norm=LogNorm())

    axes[3].axis('off')

    dish_eating_posteriors[dish_eating_posteriors < cutoff] = np.nan
    axes[4].set_title(r'$p(z_{tk}=1|o_{\leq t})$')
    sns.heatmap(dish_eating_posteriors,
                ax=axes[4],
                mask=np.isnan(dish_eating_posteriors),
                cmap='jet',
                vmin=cutoff,
                vmax=1.,
                norm=LogNorm())
    plt.savefig(os.path.join(plot_dir, 'actual_indicators_vs_inferred_indicators.png'),
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()


def plot_num_dishes_by_observation(indicators,
                                   num_dishes_poisson_rate_priors,
                                   num_dishes_poisson_rate_posteriors,
                                   plot_dir):
    seq_length = indicators.shape[0]
    obs_indices = np.arange(1 + seq_length)  # remember, we started with t = 0
    real_num_dishes = np.concatenate(
        [[0.], np.sum(np.minimum(np.cumsum(indicators, axis=0), 1), axis=1)])
    plt.plot(real_num_dishes,
             obs_indices,
             label='True')
    plt.plot(num_dishes_poisson_rate_priors[:, 0],
             obs_indices,
             label=r'$p(\Lambda_t|o_{< t})$')
    plt.plot(num_dishes_poisson_rate_posteriors[:, 0],
             obs_indices,
             label=r'$p(\Lambda_t|o_{\leq t})$')
    plt.ylabel('Customer Index')
    plt.xlabel('Number of Dishes')
    plt.legend()
    # plt.gca().axis('equal')
    plt.xlim(0, seq_length)
    plt.ylim(0, seq_length)
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(plot_dir, 'inferred_number_dishes_poisson_rates.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


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
