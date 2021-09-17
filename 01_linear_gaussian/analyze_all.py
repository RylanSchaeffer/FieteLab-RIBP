"""
Collect all subdirectories analytical and Monte Carlo estimates,
then plot MSE as function of

Example usage:

00_prior/run_mse.py
"""
import argparse
import joblib
import logging
import numpy as np
import os
import pandas as pd

import plot_linear_gaussian


def analyze_all(args: argparse.Namespace):
    # create directory
    exp_dir_path = args.exp_dir_path
    results_dir_path = os.path.join(exp_dir_path, 'results')

    results_df = load_all_datasets_all_alg_results(
        results_dir_path=results_dir_path)

    plot_linear_gaussian.plot_analyze_all_results(
        results_df=results_df,
        plot_dir=results_dir_path)


def load_all_datasets_all_alg_results(results_dir_path) -> pd.DataFrame:

    rows = []
    sampling_dirs = [subdir for subdir in os.listdir(results_dir_path)]

    # Iterate through each sampling scheme directory
    for sampling_dir in sampling_dirs:
        sampling_dir_path = os.path.join(results_dir_path, sampling_dir)
        # Iterate through each sampled dataset
        dataset_dirs = [subdir for subdir in os.listdir(sampling_dir_path)
                        if os.path.isdir(subdir)]
        for dataset_dir in dataset_dirs:
            dataset_dir_path = os.path.join(sampling_dir_path, dataset_dir)
            # Find all algorithms that were run
            inference_alg_dirs = [sub_dir for sub_dir in os.listdir(dataset_dir_path)
                                  if os.path.isdir(os.path.join(dataset_dir_path, sub_dir))]
            for inference_alg_dir in inference_alg_dirs:
                inference_alg_dir_path = os.path.join(dataset_dir_path, inference_alg_dir)
                try:
                    stored_data = joblib.load(
                        os.path.join(inference_alg_dir_path, 'inference_alg_results.joblib'))
                except FileNotFoundError:
                    continue
                new_row = [sampling_dir, dataset_dir, stored_data['inference_alg_str'],
                           stored_data['inference_alg_params']['alpha'],
                           stored_data['inference_alg_params']['beta'],
                           stored_data['runtime'],
                           stored_data['log_posterior_predictive']['mean']]
                rows.append(new_row)

    results_df = pd.DataFrame(
        rows,
        columns=['sampling', 'dataset', 'inference_alg', 'alpha',
                 'beta', 'runtime', 'log_posterior_predictive'])

    results_df['negative_log_posterior_predictive'] = -results_df['log_posterior_predictive']
    return results_df


def calc_analytical_vs_monte_carlo_mse(analytical_marginals: np.ndarray,
                                       mc_marginals: np.ndarray,
                                       num_boostraps: int = 10,
                                       num_samples: list = None) -> pd.DataFrame:
    """
    Calculate MSE between analytical marginal matrix (num obs, max num features)
    and Monte Carlo marginals matrix (num samples, num obs, max num features).
    """

    if num_samples is None:
        # num_samples = [10, 25, 100, 250, 1000, 2500]
        num_samples = [10, 100, 1000]
    numsample_meanerror_semerror = []
    for num_sample in num_samples:
        for boostrap_idx in range(num_boostraps):
            # draw sample from CRP
            rand_indices = np.random.choice(np.arange(mc_marginals.shape[0]),
                                            size=num_sample)
            mean_mc_marginals_of_rand_indices = np.mean(
                mc_marginals[rand_indices],
                axis=0)
            mse = np.square(np.linalg.norm(
                analytical_marginals - mean_mc_marginals_of_rand_indices))
            numsample_meanerror_semerror.append([num_sample, boostrap_idx, mse])
    return numsample_meanerror_semerror


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir_path', type=str,
                        default='01_linear_gaussian',
                        help='Path to write plots and other results to.')
    args = parser.parse_args()
    analyze_all(args=args)
    logging.info('Finished.')
