from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict, List

import utils.plot_general



plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16  # was previously 22
sns.set_style("whitegrid")


def plot_analyze_all_algorithms_results(inf_algorithms_results_df: pd.DataFrame,
                                        inf_algs_num_features_by_num_obs: List[np.ndarray],
                                        plot_dir: str):

    alphas = inf_algorithms_results_df['alpha'].unique()
    betas = inf_algorithms_results_df['beta'].unique()

    for alpha, beta in product(alphas, betas):
        indices = inf_algorithms_results_df[
            (inf_algorithms_results_df['alpha'] == alpha) & (inf_algorithms_results_df['beta'] == beta)
        ].index.values
        alpha_beta_num_features_by_num_obs = np.stack([
            inf_algs_num_features_by_num_obs[idx] for idx in indices])
        avg_alpha_beta_num_features_by_num_obs = np.mean(
            alpha_beta_num_features_by_num_obs,
            axis=0)
        plt.plot(1 + np.arange(len(avg_alpha_beta_num_features_by_num_obs)),
                 avg_alpha_beta_num_features_by_num_obs,
                 label=rf'$\alpha={alpha}, \beta={beta}$',)
    # plt.legend()
    plt.xlabel('Number of Observations')
    plt.ylabel('Number of Features')
    plt.savefig(os.path.join(plot_dir,
                             'num_features_by_num_obs_groupedby_alpha_beta.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

    # for num_features_by_num_obs in inf_algs_num_features_by_num_obs:
    #     plt.plot(1 + np.arange(len(num_features_by_num_obs)),
    #              num_features_by_num_obs,
    #              color='k',
    #              alpha=0.1,
    #              markeredgewidth=2)
    # plt.show()
    # print(10)

    # utils.plot.plot_neg_log_posterior_predictive_by_alpha_beta(
    #     inf_algorithms_results_df=inf_algorithms_results_df,
    #     plot_dir=plot_dir)


def plot_run_one_inference_results(sampled_omniglot_data: Dict,
                                   inference_alg_results: Dict,
                                   inference_alg_str: str,
                                   inference_alg_params: Dict[str, Dict[str, float]],
                                   log_posterior_predictive_dict: Dict,
                                   plot_dir: str):
    # utils.plot.plot_run_one_indicators_by_num_obs(
    #     dish_eating_priors=inference_alg_results['dish_eating_priors'],
    #     dish_eating_posteriors=inference_alg_results['dish_eating_posteriors'],
    #     indicators=None,
    #     plot_dir=plot_dir)

    plot_images_belonging_to_top_k_features(
        inference_alg_str=inference_alg_str,
        inference_alg_params=inference_alg_params,
        images=sampled_omniglot_data['images'],
        dish_eating_posteriors=inference_alg_results['dish_eating_posteriors'],
        dish_eating_posteriors_running_sum=inference_alg_results['dish_eating_posteriors_running_sum'],
        plot_dir=plot_dir,
    )

    utils.plot.plot_run_one_num_features_by_num_obs_using_poisson_rates(
        num_dishes_poisson_rate_priors=inference_alg_results['num_dishes_poisson_rate_priors'],
        num_dishes_poisson_rate_posteriors=inference_alg_results['num_dishes_poisson_rate_posteriors'],
        indicators=None,
        plot_dir=plot_dir)


def plot_images_belonging_to_top_k_features(inference_alg_str: str,
                                            inference_alg_params: Dict[str, Dict[str, float]],
                                            images: np.ndarray,
                                            dish_eating_posteriors: np.ndarray,
                                            dish_eating_posteriors_running_sum: np.ndarray,
                                            plot_dir):
    # as a heuristic for plotting
    confident_dish_eating_posteriors = dish_eating_posteriors > 0.4
    summed_confident_dish_eating_posteriors = np.sum(
        confident_dish_eating_posteriors,
        axis=0)

    # plt.plot(dish_indices, table_assignment_posteriors_running_sum[-1, :], label='Total Prob. Mass')
    # plt.plot(dish_indices, summed_confident_dish_eating_posteriors, label='Confident Predictions')
    # plt.ylabel('Prob. Mass at Table')
    # plt.xlabel('Table Index')
    # plt.xlim(0, 150)
    # plt.legend()
    #
    # plt.savefig(os.path.join(plot_dir,
    #                          '{}_alpha={:.2f}_mass_per_table.png'.format(inference_alg_str,
    #                                                                      concentration_param)),
    #             bbox_inches='tight',
    #             dpi=300)
    # plt.show()
    # plt.close()

    dish_indices_by_decreasing_summed_prob_mass = np.argsort(
        summed_confident_dish_eating_posteriors)[::-1]

    num_rows = num_features = 6
    max_num_images_per_row = 11
    num_cols = max_num_images_per_row
    fig, axes = plt.subplots(nrows=num_rows,
                             ncols=num_cols,
                             sharex=True,
                             sharey=True)
    # axes[0, 0].set_title(f'Cluster Means')
    # axes[0, int(max_num_images_per_row / 2)].set_title('Observations')

    for feature_idx in range(num_features):

        dish_idx = dish_indices_by_decreasing_summed_prob_mass[feature_idx]

        axes[feature_idx, 0].set_ylabel(f'Feature: {1 + feature_idx}',
                                        rotation=0,
                                        labelpad=40)
        # axes[row_idx, 0].axis('off')

        # Identify observations with highest presence of feature
        posteriors_at_table = dish_eating_posteriors[:, dish_idx]
        customer_indices_by_decreasing_prob_mass = np.argsort(
            posteriors_at_table)[::-1]

        for image_num in range(max_num_images_per_row):
            customer_idx = customer_indices_by_decreasing_prob_mass[image_num]
            customer_mass = posteriors_at_table[customer_idx]
            # only plot high confidence
            try:
                if customer_mass < 0.4:
                    axes[feature_idx, image_num].axis('off')
                else:
                    axes[feature_idx, image_num].imshow(images[customer_idx], cmap='gray')
                    axes[feature_idx, image_num].set_title(f'{np.round(customer_mass, 2)}')
            except IndexError:
                axes[feature_idx, image_num].axis('off')

    # remove tick labels
    plt.setp(axes, xticks=[], yticks=[])
    plt.savefig(os.path.join(plot_dir,
                             'images_belonging_to_top_k_features.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
