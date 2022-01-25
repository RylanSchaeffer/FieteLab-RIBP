"""
Iterate through all the generated synthetic data and the inference
algorithms' results, and plot performance (e.g. log posterior predictive
vs runtime).

Example usage:

03_omniglot/analyze_all.py
"""
import argparse
import joblib
import logging
import numpy as np
import os
import pandas as pd
from typing import List, Tuple

import plot_omniglot


def analyze_all(args: argparse.Namespace):
    # create directory
    exp_dir_path = args.exp_dir_path
    results_dir_path = os.path.join(exp_dir_path, 'results')

    inf_algs_results_df, inf_algs_num_features_by_num_obs = load_all_inf_alg_results(
        results_dir_path=results_dir_path)

    inf_algs_results_df.sort_values(
        by=['log_posterior_predictive'],
        inplace=True,
        ascending=False)

    inf_algs_results_df.to_csv(
        os.path.join(exp_dir_path, 'inf_algorithms_results_df.csv'),
        index=False)

    plot_omniglot.plot_analyze_all_algorithms_results(
        inf_algorithms_results_df=inf_algs_results_df,
        inf_algs_num_features_by_num_obs=inf_algs_num_features_by_num_obs,
        plot_dir=exp_dir_path)


def load_all_inf_alg_results(results_dir_path: str,
                             ) -> Tuple[pd.DataFrame, List[np.ndarray]]:

    inf_algorithms_results_rows = []
    inf_algorithms_num_features_by_num_obs = []
    run_dirs = [subdir for subdir in os.listdir(results_dir_path)]

    # Iterate through each sampling scheme directory
    for run_dir in run_dirs:
        run_dir_path = os.path.join(results_dir_path, run_dir)
        try:
            stored_data = joblib.load(
                os.path.join(run_dir_path, 'inference_alg_results.joblib'))
        except FileNotFoundError:
            logging.info(f'Could not find results for {run_dir_path}.')
            continue
        inf_algorithms_results_row = [
            stored_data['inference_alg_str'],
            stored_data['inference_alg_params']['IBP']['alpha'],
            stored_data['inference_alg_params']['IBP']['beta'],
            stored_data['inference_alg_params']['feature_prior_params']['feature_prior_cov_scaling'],
            stored_data['inference_alg_params']['scale_prior_params']['scale_prior_cov_scaling'],
            stored_data['inference_alg_params']['likelihood_params']['sigma_x'],
            stored_data['runtime'],
            stored_data['log_posterior_predictive']['mean']]

        inf_algorithms_results_rows.append(inf_algorithms_results_row)

        num_features_by_num_obs = stored_data['inference_alg_results'][
                                      'num_dishes_poisson_rate_posteriors'][:, 0]  # remove extra dimension
        inf_algorithms_num_features_by_num_obs.append(num_features_by_num_obs)

        del stored_data

    inf_algorithms_results_df = pd.DataFrame(
        inf_algorithms_results_rows,
        columns=['inference_alg', 'alpha', 'beta', 'feature_cov_scaling',
                 'scale_cov_scaling', 'likelihood_cov_scaling', 'runtime',
                 'log_posterior_predictive'])

    inf_algorithms_results_df['negative_log_posterior_predictive'] = \
        -inf_algorithms_results_df['log_posterior_predictive']

    return inf_algorithms_results_df, inf_algorithms_num_features_by_num_obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir_path', type=str,
                        default='03_omniglot',
                        help='Path to write plots and other results to.')
    args = parser.parse_args()
    analyze_all(args=args)
    logging.info('Finished.')
