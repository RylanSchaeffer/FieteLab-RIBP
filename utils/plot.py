# Common plotting functions
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from utils.numpy_helpers import compute_largest_dish_idx


def plot_run_one_indicators_by_num_obs(indicators: np.ndarray,
                                       dish_eating_priors,
                                       dish_eating_posteriors,
                                       plot_dir,
                                       cutoff: float = 1e-2):
    """
    Plot three heatmaps together. The left is the ground truth indicators (y)
    vs observation index (x). The middle is the prior over indicators and the
    right is posterior over indicators.

    The first three 3 inputs are expected to have shape
    (number of obs, number indicators)
    """

    num_obs = indicators.shape[0]
    yticklabels = 1 + np.arange(num_obs)
    fig, axes = plt.subplots(nrows=1,
                             ncols=5,
                             figsize=(12, 5),
                             gridspec_kw={'width_ratios': [1, 0.25, 1, 0.25, 1]})

    # first figure out the largest dish index with a value greater
    largest_indicator_idx = compute_largest_dish_idx(
        observations=indicators,
        cutoff=cutoff)
    largest_dish_prior_idx = compute_largest_dish_idx(
        observations=dish_eating_priors,
        cutoff=cutoff)
    largest_dish_posterior_idx = compute_largest_dish_idx(
        observations=dish_eating_posteriors,
        cutoff=cutoff)
    # take mean of the three
    max_feature_idx_to_display = int(np.mean([
        largest_indicator_idx, largest_dish_prior_idx, largest_dish_posterior_idx]))
    xticklabels = 1 + np.arange(max_feature_idx_to_display)

    indicators = indicators.astype(float)
    possibly_bigger_indicators = np.zeros(
        shape=(num_obs, max(max_feature_idx_to_display,
                            indicators.shape[1])))
    possibly_bigger_indicators[:, :indicators.shape[1]] = indicators[:, :]
    possibly_bigger_indicators[possibly_bigger_indicators < cutoff] = np.nan
    axes[0].set_title(r'$z_{nk}$')
    sns.heatmap(possibly_bigger_indicators,
                ax=axes[0],
                mask=np.isnan(possibly_bigger_indicators),
                cmap='jet',
                vmin=cutoff,
                vmax=1.,
                norm=LogNorm(),
                xticklabels=xticklabels,
                yticklabels=yticklabels)
    axes[0].set_ylabel('Observation Index')
    axes[0].set_xlabel('Feature Index')

    axes[1].axis('off')

    dish_eating_priors[dish_eating_priors < cutoff] = np.nan
    axes[2].set_title(r'$p(z_{nk}=1|o_{<n})$')
    sns.heatmap(dish_eating_priors[:, :max_feature_idx_to_display],
                ax=axes[2],
                mask=np.isnan(dish_eating_priors[:, :max_feature_idx_to_display]),
                cmap='jet',
                vmin=cutoff,
                vmax=1.,
                norm=LogNorm(),
                xticklabels=xticklabels,
                yticklabels=yticklabels)
    axes[2].set_ylabel('Observation Index')
    axes[2].set_xlabel('Feature Index')
    axes[3].axis('off')

    dish_eating_posteriors[dish_eating_posteriors < cutoff] = np.nan
    axes[4].set_title(r'$p(z_{nk}=1|o_{\leq n})$')
    sns.heatmap(dish_eating_posteriors[:, :max_feature_idx_to_display],
                ax=axes[4],
                mask=np.isnan(dish_eating_posteriors[:, :max_feature_idx_to_display]),
                cmap='jet',
                vmin=cutoff,
                vmax=1.,
                norm=LogNorm(),
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                )
    axes[4].set_ylabel('Observation Index')
    axes[4].set_xlabel('Feature Index')
    plt.savefig(os.path.join(plot_dir, 'indicators_by_num_obs.png'),
                bbox_inches='tight',
                dpi=500)
    # plt.show()
    plt.close()


def plot_indicators_by_index_over_many_observations(indicators, ):
    # plt.title(f'Obs Idx: {obs_idx}, VI Idx: {vi_idx + 1}')
    # plt.scatter(np.arange(len(dish_eating_prior)), dish_eating_prior, label='prior')
    # plt.scatter(np.arange(len(dish_eating_prior)), Z_probs.detach().numpy(), label='posterior')
    # plt.legend()
    # plt.xlim(0, 15)
    # plt.show()
    raise NotImplementedError


def plot_inference_algs_comparison(inference_algs_results_by_dataset_idx: dict,
                                   dataset_by_dataset_idx: dict,
                                   plot_dir: str):
    num_datasets = len(inference_algs_results_by_dataset_idx)
    num_clusters = len(np.unique(dataset_by_dataset_idx[0]['assigned_table_seq']))

    inference_algs = list(inference_algs_results_by_dataset_idx[0].keys())
    scoring_metrics = inference_algs_results_by_dataset_idx[0][inference_algs[0]]['scores_by_param'].columns.values

    # we have four dimensions of interest: inference_alg, dataset idx, scoring metric, concentration parameter

    # construct dictionary mapping from inference alg to dataframe
    # with dataset idx as rows and concentration params as columns
    # {inference alg: DataFrame(number of clusters)}
    num_clusters_by_dataset_by_inference_alg = {}
    for inference_alg in inference_algs:
        num_clusters_by_dataset_by_inference_alg[inference_alg] = pd.DataFrame([
            inference_algs_results_by_dataset_idx[dataset_idx][inference_alg]['num_clusters_by_param']
            for dataset_idx in range(num_datasets)])

    plot_inference_algs_num_clusters_by_param(
        num_clusters_by_dataset_by_inference_alg=num_clusters_by_dataset_by_inference_alg,
        plot_dir=plot_dir,
        num_clusters=num_clusters)

    # construct dictionary mapping from inference alg to dataframe
    # with dataset idx as rows and concentration params as columns
    # {inference alg: DataFrame(runtimes)}
    runtimes_by_dataset_by_inference_alg = {}
    for inference_alg in inference_algs:
        runtimes_by_dataset_by_inference_alg[inference_alg] = pd.DataFrame([
            inference_algs_results_by_dataset_idx[dataset_idx][inference_alg]['runtimes_by_param']
            for dataset_idx in range(num_datasets)])

    plot_inference_algs_runtimes_by_param(
        runtimes_by_dataset_by_inference_alg=runtimes_by_dataset_by_inference_alg,
        plot_dir=plot_dir)

    # construct dictionary mapping from scoring metric to inference alg
    # to dataframe with dataset idx as rows and concentration params as columns
    # {scoring metric: {inference alg: DataFrame(scores)}}
    scores_by_dataset_by_inference_alg_by_scoring_metric = {}
    for scoring_metric in scoring_metrics:
        scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric] = {}
        for inference_alg in inference_algs:
            scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric][inference_alg] = \
                pd.DataFrame(
                    [inference_algs_results_by_dataset_idx[dataset_idx][inference_alg]['scores_by_param'][
                         scoring_metric]
                     for dataset_idx in range(num_datasets)])

    plot_inference_algs_scores_by_param(
        scores_by_dataset_by_inference_alg_by_scoring_metric=scores_by_dataset_by_inference_alg_by_scoring_metric,
        plot_dir=plot_dir)


def plot_inference_algs_num_clusters_by_param(num_clusters_by_dataset_by_inference_alg: dict,
                                              plot_dir: str,
                                              num_clusters: int):
    for inference_alg_str, inference_alg_num_clusters_df in num_clusters_by_dataset_by_inference_alg.items():
        means = inference_alg_num_clusters_df.mean()
        sems = inference_alg_num_clusters_df.sem()
        plt.plot(inference_alg_num_clusters_df.columns.values,  # concentration params
                 means,
                 label=inference_alg_str)
        plt.fill_between(
            x=inference_alg_num_clusters_df.columns.values,
            y1=means - sems,
            y2=means + sems,
            alpha=0.3,
            linewidth=0)

    plt.xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
    plt.ylabel('Number of Clusters')
    plt.axhline(num_clusters, label='Correct Number Clusters', linestyle='--', color='k')
    plt.gca().set_ylim(bottom=1.)
    plt.gca().set_xlim(left=0)
    plt.legend()
    plt.yscale('log')
    plt.savefig(os.path.join(plot_dir, f'num_clusters_by_param.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_inference_algs_scores_by_param(scores_by_dataset_by_inference_alg_by_scoring_metric: dict,
                                        plot_dir: str):
    # for each scoring function, plot score (y) vs parameter (x)
    for scoring_metric, scores_by_dataset_by_inference_alg in scores_by_dataset_by_inference_alg_by_scoring_metric.items():
        for inference_alg_str, inference_algs_scores_df in scores_by_dataset_by_inference_alg.items():
            plt.plot(inference_algs_scores_df.columns.values,  # concentration params
                     inference_algs_scores_df.mean(),
                     label=inference_alg_str)
            plt.fill_between(
                x=inference_algs_scores_df.columns.values,
                y1=inference_algs_scores_df.mean() - inference_algs_scores_df.sem(),
                y2=inference_algs_scores_df.mean() + inference_algs_scores_df.sem(),
                alpha=0.3,
                linewidth=0)

        plt.legend()
        plt.xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
        plt.ylabel(scoring_metric)
        plt.ylim(0., 1.)
        plt.gca().set_xlim(left=0)
        plt.savefig(os.path.join(plot_dir, f'comparison_score={scoring_metric}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_inference_algs_runtimes_by_param(runtimes_by_dataset_by_inference_alg: dict,
                                          plot_dir: str):
    for inference_alg_str, inference_alg_runtime_df in runtimes_by_dataset_by_inference_alg.items():
        print(f'{inference_alg_str} Average Runtime: {np.mean(inference_alg_runtime_df.values)}')
        means = inference_alg_runtime_df.mean()
        sems = inference_alg_runtime_df.sem()
        plt.plot(inference_alg_runtime_df.columns.values,  # concentration params
                 means,
                 label=inference_alg_str)
        plt.fill_between(
            x=inference_alg_runtime_df.columns.values,
            y1=means - sems,
            y2=means + sems,
            alpha=0.3,
            linewidth=0)

    plt.xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
    plt.ylabel('Runtime (s)')
    plt.gca().set_xlim(left=0)
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'runtimes_by_param.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_run_one_num_features_by_num_obs(indicators,
                                         num_dishes_poisson_rate_priors,
                                         num_dishes_poisson_rate_posteriors,
                                         plot_dir):
    """
    Plot inferred Poisson rates of number of dishes (Y) vs number of observations (X).
    """
    seq_length = indicators.shape[0]
    obs_indices = 1 + np.arange(seq_length)  # remember, we started with t = 0
    real_num_dishes = np.concatenate(
        [np.sum(np.minimum(np.cumsum(indicators, axis=0), 1), axis=1)])

    # r'$q(\Lambda_t|o_{< t})$'
    # r'$q(\Lambda_t|o_{\leq t})$'
    data_to_plot = pd.DataFrame.from_dict({
        'obs_idx': obs_indices,
        'True': real_num_dishes,
        'Prior': num_dishes_poisson_rate_priors[:, 0],
        'Posterior': num_dishes_poisson_rate_posteriors[:, 0],
    })

    melted_data_to_plot = data_to_plot.melt(
        id_vars=['obs_idx'],  # columns to keep
        var_name='quantity',  # new column name for previous columns' headers
        value_name='num_dishes',  # new column name for values
    )

    g = sns.lineplot(x='obs_idx', y='num_dishes', data=melted_data_to_plot,
                     hue='quantity')

    # Remove "quantity" from legend title
    # see https://stackoverflow.com/questions/51579215/remove-seaborn-lineplot-legend-title
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles[1:], labels=labels[1:])

    # remove quantity from title
    plt.xlabel('Number of Observations')
    plt.ylabel('Inferred Number of Features')
    plt.ylim(bottom=0.)
    # plt.gca().axis('equal')
    # plt.xlim(0, seq_length)
    # plt.ylim(0, seq_length)
    # plt.gca().invert_yaxis()
    g.get_legend().set_title(None)
    plt.savefig(os.path.join(plot_dir, 'num_features_by_num_obs.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

# def plot_posterior_num_dishes_by_num_obs(dish_eating_array: np.ndarray,
#                                          plot_dir: str,
#                                          cutoff: float = 1e-5):
#     """
#     Plot number of dishes (Y) vs number of observations (X).
#
#     Assumes dish_eating_array has shape (num obs, max num indicators)
#     """
#
#     cum_dish_eating_array = np.cumsum(dish_eating_array, axis=0)
#     num_dishes_by_num_obs = np.sum(cum_dish_eating_array > cutoff, axis=1)
#
#     obs_indices = 1 + np.arange(len(num_dishes_by_num_obs))
#
#     sns.lineplot(obs_indices,
#                  num_dishes_by_num_obs)
#     plt.xlabel('Number of Observations')
#     plt.ylabel(f'Total Number of Dishes (cutoff={cutoff})')
#     plt.legend()
#
#     # make axes equal
#     plt.axis('square')
#
#     plt.savefig(os.path.join(plot_dir, 'num_dishes_by_num_obs.png'),
#                 bbox_inches='tight',
#                 dpi=300)
#     # plt.show()
#     plt.close()
