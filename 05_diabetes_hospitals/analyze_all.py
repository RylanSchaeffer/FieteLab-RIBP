"""
Load the inference algorithms' results on the 2016 cancer gene expression
dataset, and plot performance (e.g. log posterior predictive vs runtime).

Example usage:

05_diabetes_hospitals/analyze_all.py
"""
import argparse
import copy
import joblib
import logging
import numpy as np
import os
import pandas as pd
from typing import List, Tuple

import plot_diabetes_hospitals


def analyze_all(args: argparse.Namespace):
    # create directory
    exp_dir_path = args.exp_dir_path
    results_dir_path = os.path.join(exp_dir_path, 'results_data=100')
    plot_dir = os.path.join(results_dir_path, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    inf_algs_results_df = load_all_inf_alg_results(
        results_dir_path=results_dir_path)

    inf_algs_results_df.to_csv(
        os.path.join(results_dir_path, 'inf_algs_results_df.csv'),
        index=False)

    plot_diabetes_hospitals.plot_analyze_all_algorithms_results(
        inf_algs_results_df=inf_algs_results_df,
        plot_dir=plot_dir)


def load_all_inf_alg_results(results_dir_path: str,
                             ) -> pd.DataFrame:
    inf_algorithms_results_rows = []
    # inf_algorithms_num_features_by_num_obs = []
    run_dirs = [subdir for subdir in os.listdir(results_dir_path)]

    for run_dir in run_dirs:
        run_dir_path = os.path.join(results_dir_path, run_dir)
        if not os.path.isdir(run_dir_path):
            continue
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
            stored_data['inference_alg_params']['likelihood_params']['sigma_x'],
            stored_data['runtime'],
            stored_data['log_posterior_predictive']['mean'],
            # stored_data['reconstruction_error'],
        ]

        # Copy to ensure we don't keep any references to stored_data.
        # Was having memory issues otherwise.
        inf_algorithms_results_rows.append(copy.deepcopy(inf_algorithms_results_row))

        # num_features_by_num_obs = stored_data['inference_alg_results'][
        #                               'num_dishes_poisson_rate_posteriors'][:, 0]  # remove extra dimension
        # inf_algorithms_num_features_by_num_obs.append(num_features_by_num_obs)

        del stored_data

        print(f'Loaded {run_dir_path}')

    inf_algs_results_df = pd.DataFrame(
        inf_algorithms_results_rows,
        columns=['inference_alg', 'alpha', 'beta', 'feature_cov_scaling',
                 'likelihood_cov_scaling', 'runtime',
                 'log_posterior_predictive',
                 # 'reconstruction_error',
                 ])

    inf_algs_results_df['negative_log_posterior_predictive'] = \
        -inf_algs_results_df['log_posterior_predictive']

    return inf_algs_results_df  # , inf_algorithms_num_features_by_num_obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir_path', type=str,
                        default='05_diabetes_hospitals',
                        help='Path to write plots and other results to.')
    args = parser.parse_args()
    analyze_all(args=args)
    logging.info('Finished.')
