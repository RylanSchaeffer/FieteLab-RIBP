"""
Compare the analytical RCRP marginal distribution against
Monte-Carlo estimates of the RCRP marginal distribution.

Example usage:

00_prior/run_one.py --results_dir_path=00_prior/results \
 --num_customer=50 --num_mc_sample=2000 \
 --alpha=2.51 --beta=3.7
"""

import argparse
import logging
import joblib
import numpy as np
import os
import scipy.stats
from typing import Dict


import plot_prior
import utils.data_synthetic
import utils.run_helpers


def run_one(args: argparse.Namespace):

    run_one_results_dir = setup(args=args)

    sample_ibp_results = sample_ibp_and_save(
        num_customer=args.num_customer,
        alpha=args.alpha,
        beta=args.beta,
        run_one_results_dir=run_one_results_dir,
        num_mc_sample=args.num_mc_sample)

    analytical_ibp_results = compute_analytical_ibp_and_save(
        num_customer=args.num_customer,
        alpha=args.alpha,
        beta=args.beta,
        run_one_results_dir=run_one_results_dir)

    plot_prior.plot_customer_dishes_analytical_vs_monte_carlo(
        sampled_dishes_by_customer_idx=sample_ibp_results['sampled_dishes_by_customer_idx'],
        analytical_dishes_by_customer_idx=analytical_ibp_results['analytical_dishes_by_customer_idx'],
        alpha=args.alpha,
        beta=args.beta,
        plot_dir=run_one_results_dir)

    plot_prior.plot_num_dishes_analytical_vs_monte_carlo(
        sampled_num_dishes_by_customer_idx=sample_ibp_results['num_dishes_by_customer_idx'],
        analytical_num_dishes_by_customer=analytical_ibp_results['num_dishes_by_customer_idx'],
        alpha=args.alpha,
        beta=args.beta,
        plot_dir=run_one_results_dir)

    plot_prior.plot_recursion_visualization(
        cum_analytical_dishes_by_customer_idx=analytical_ibp_results['cum_analytical_dishes_by_customer_idx'],
        analytical_num_dishes_by_customer=analytical_ibp_results['num_dishes_by_customer_idx'],
        analytical_dishes_by_customer_idx=analytical_ibp_results['analytical_dishes_by_customer_idx'],
        alpha=args.alpha,
        beta=args.beta,
        plot_dir=run_one_results_dir)


def compute_analytical_ibp(num_customer: int,
                           alpha: float,
                           beta: float) -> Dict[str, np.ndarray]:

    """

    :param num_customer:
    :param alpha:
    :param beta:
    """

    assert alpha > 0.
    assert beta > 0.

    # Create arrays to store all information.
    # To make Python indexing match the mathematical notation, we'll use 1-based
    # indexing and then cut off the extra row and column at the end.
    # preallocate results

    # use 10 * expected number of dishes as heuristic
    # needs to match whatever the sampled IBP heuristic is, otherwise shapes agree
    max_dishes = 10 * int(alpha * beta * np.sum(1 / (1 + np.arange(num_customer))))
    analytical_dishes_by_customer_idx = np.zeros(
        shape=(num_customer+1, max_dishes+1),
        dtype=np.float32)
    cum_analytical_dishes_by_customer_idx = np.zeros(
        shape=(num_customer+1, max_dishes+1),
        dtype=np.float32)
    num_dishes_by_customer_idx = np.zeros(
        shape=(num_customer+1, max_dishes + 1),
        dtype=np.float32)

    dish_indices = np.arange(max_dishes + 1)

    # customer 1 samples only new dishes
    new_dishes_rate = alpha * beta / (beta + 1 - 1)
    total_dishes_rate = new_dishes_rate
    num_dishes_by_customer_idx[1, :] = scipy.stats.poisson.pmf(
            dish_indices, mu=total_dishes_rate)
    analytical_dishes_by_customer_idx[1, :] = np.cumsum(
        scipy.stats.poisson.pmf(dish_indices[::-1], mu=new_dishes_rate))[::-1]
    cum_analytical_dishes_by_customer_idx[1, :] = analytical_dishes_by_customer_idx[1, :]

    # all subsequent customers sample new dishes
    for cstmr_idx in range(2, num_customer + 1):
        analytical_dishes_by_customer_idx[cstmr_idx, :] = \
            cum_analytical_dishes_by_customer_idx[cstmr_idx - 1, :] / (beta + cstmr_idx - 1)
        new_dishes_rate = alpha * beta / (beta + cstmr_idx - 1)
        cdf_lambda_t_minus_1 = scipy.stats.poisson.cdf(dish_indices, mu=total_dishes_rate)
        cdf_lambda_t = scipy.stats.poisson.cdf(dish_indices, mu=total_dishes_rate + new_dishes_rate)
        cdf_diff = np.subtract(cdf_lambda_t_minus_1, cdf_lambda_t)
        analytical_dishes_by_customer_idx[cstmr_idx, :] += cdf_diff
        cum_analytical_dishes_by_customer_idx[cstmr_idx, :] = np.add(
            cum_analytical_dishes_by_customer_idx[cstmr_idx - 1, :],
            analytical_dishes_by_customer_idx[cstmr_idx, :])
        total_dishes_rate += new_dishes_rate
        print(total_dishes_rate)
        num_dishes_by_customer_idx[cstmr_idx, :] = scipy.stats.poisson.pmf(
            dish_indices, mu=total_dishes_rate)

    # Cutoff extra row and columns we introduced at the beginning.
    analytical_dishes_by_customer_idx = analytical_dishes_by_customer_idx[1:, 1:]
    cum_analytical_dishes_by_customer_idx = cum_analytical_dishes_by_customer_idx[1:, 1:]
    num_dishes_by_customer_idx = num_dishes_by_customer_idx[1:, 1:]

    # import matplotlib.pyplot as plt
    # plt.imshow(num_dishes_by_customer_idx[:, :50])
    # plt.show()


    analytical_dcrp_results = {
        'cum_analytical_dishes_by_customer_idx': cum_analytical_dishes_by_customer_idx,
        'analytical_dishes_by_customer_idx': analytical_dishes_by_customer_idx,
        'num_dishes_by_customer_idx': num_dishes_by_customer_idx,
    }

    return analytical_dcrp_results


def compute_analytical_ibp_and_save(num_customer: int,
                                    alpha: float,
                                    beta: float,
                                    run_one_results_dir: str) -> Dict[str, np.ndarray]:

    crp_analytical_path = os.path.join(
        run_one_results_dir,
        'analytical.joblib')

    # if not os.path.isfile(crp_analytical_path):
    analytical_dcrp_results = compute_analytical_ibp(
        num_customer=num_customer,
        alpha=alpha,
        beta=beta)

    logging.info(f'Computed analytical results for {crp_analytical_path}')
    joblib.dump(filename=crp_analytical_path,
                value=analytical_dcrp_results)

    # this gives weird error: joblib ValueError: EOF: reading array data, expected 262144 bytes got 225056
    # analytical_dcrp_results = joblib.load(crp_analytical_path)

    logging.info(f'Loaded analytical results for {crp_analytical_path}')
    return analytical_dcrp_results


def sample_ibp_and_save(num_customer: int,
                        alpha: float,
                        beta: float,
                        num_mc_sample: int,
                        run_one_results_dir: str) -> Dict[str, np.ndarray]:

    ibp_samples_path = os.path.join(
        run_one_results_dir,
        f'monte_carlo_samples={num_mc_sample}.joblib')

    # if not os.path.isfile(ibp_samples_path):

    sample_ibp_results = utils.data_synthetic.sample_ibp(
        num_mc_sample=num_mc_sample,
        num_customer=num_customer,
        alpha=alpha,
        beta=beta)
    logging.info(f'Generated samples for {ibp_samples_path}')
    joblib.dump(filename=ibp_samples_path,
                value=sample_ibp_results)
    logging.info(f'Loaded samples for {ibp_samples_path}')
    return sample_ibp_results


def setup(args: argparse.Namespace):
    run_one_results_dir = os.path.join(
        args.results_dir_path,
        f'a={args.alpha}_b={args.beta}')
    os.makedirs(run_one_results_dir, exist_ok=True)

    utils.run_helpers.create_logger(run_dir=run_one_results_dir)

    np.random.seed(args.seed)
    return run_one_results_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--results_dir_path', type=str,
                        help='Path to write plots and other results to.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Pseudo-random seed for NumPy/PyTorch.')
    parser.add_argument('--num_customer', type=int,
                        help='Number of customers per Monte Carlo sample.')
    parser.add_argument('--num_mc_sample', type=int,
                        help='Number of Monte Carlo samples from conditional.')
    parser.add_argument('--alpha', type=float,
                        help='IBP alpha parameter.')
    parser.add_argument('--beta', type=float,
                        help='IBP beta parameter.')
    args = parser.parse_args()
    run_one(args)
    logging.info('Finished.')
