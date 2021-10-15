import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from typing import Dict

import pandas as pd

import utils.plot


def plot_analyze_all_algorithms_results(inf_algorithms_results_df: pd.DataFrame,
                                        plot_dir: str):

    utils.plot.plot_neg_log_posterior_predictive_by_alpha_beta(
        inf_algorithms_results_df=inf_algorithms_results_df,
        plot_dir=plot_dir)


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
    axes[0, int(max_num_images_per_row / 2)].set_title('Observations')

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
                if customer_mass < 0.7:
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
