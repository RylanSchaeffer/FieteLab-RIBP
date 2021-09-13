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
import scipy.stats

import plot_prior


def run_mse(args: argparse.Namespace):

    # create directory
    exp_dir_path = '00_prior'
    results_dir_path = os.path.join(exp_dir_path, 'results')

    num_boostraps = args.num_bootstraps
    rows = []
    for subdir in os.listdir(results_dir_path):
        alpha, beta = subdir.split('_')
        try:
            analytical_results = joblib.load(os.path.join(results_dir_path, subdir, 'analytical.joblib'))
            mc_results = joblib.load(os.path.join(results_dir_path, subdir, 'monte_carlo_samples=5000.joblib'))
        except FileNotFoundError:
            continue

        alpha = float(alpha[2:])
        beta = float(beta[2:])
        logging.info('')
        numsample_meanerror_semerror = calc_analytical_vs_monte_carlo_mse(
            analytical_marginals=analytical_results['analytical_dishes_by_customer_idx'],
            mc_marginals=mc_results['sampled_dishes_by_customer_idx'],
            num_boostraps=num_boostraps)
        # add alpha, beta to list of [num_samples, mean_error, sem_error]
        for entry in numsample_meanerror_semerror:
            entry.extend([alpha, beta])
        rows.extend(numsample_meanerror_semerror)

    mse_df = pd.DataFrame(
        rows,
        columns=['num_samples', 'bootstrap_idx', 'mse', 'alpha', 'beta'])

    plot_prior.plot_analytical_vs_monte_carlo_mse(
        mse_df=mse_df,
        plot_dir=results_dir_path)


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
    parser.add_argument('--num_bootstraps', type=int, default=10,
                        help='Number of bootstraps to draw.')
    args = parser.parse_args()
    run_mse(args=args)
    logging.info('Finished.')
