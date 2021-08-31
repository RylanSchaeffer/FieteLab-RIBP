"""
Perform inference in a linear-Gaussian model X = ZA + epsilon
for the specified inference algorithm.

Example usage:

02_griffiths_ghahramani/run_one.py
 --run_one_results_dir=02_griffiths_ghahramani/results/categorical_probs=[0.5,0.5,0.5,0.5]/dataset=0 \
 --inference_alg_str=R-IBP --alpha=5.91 --beta=4.3
"""

import argparse
import joblib
import logging
import numpy as np
import os
from timeit import default_timer as timer
import torch

import plot_linear_gaussian
import utils.data
import utils.helpers
import utils.inference
import utils.metrics


def run_one(args: argparse.Namespace):

    setup_results = setup(args=args)

    logging.info('Running and plotting {} with params {} on dataset {}'.format(
        setup_results['inference_alg_str'],
        setup_results['inference_alg_params'],
        args.run_one_results_dir))

    run_and_plot_inference_alg(
        sampled_linear_gaussian_data=setup_results['sampled_linear_gaussian_data'],
        inference_alg_str=setup_results['inference_alg_str'],
        inference_alg_params=setup_results['inference_alg_params'],
        inference_results_dir=setup_results['inference_results_dir'])

    logging.info('Successfully ran and plotted {} with params {} on dataset {}'.format(
        setup_results['inference_alg_str'],
        setup_results['inference_alg_params'],
        args.run_one_results_dir))


def run_and_plot_inference_alg(sampled_linear_gaussian_data,
                               inference_alg_str,
                               inference_alg_params,
                               inference_results_dir):

    inference_results_path = os.path.join(
        inference_results_dir,
        'inference_alg_results.joblib')

    # run inference algorithm
    # time using timer because https://stackoverflow.com/a/25823885/4570472
    start_time = timer()
    inference_alg_results = utils.inference.run_inference_alg(
        inference_alg_str=inference_alg_str,
        observations=sampled_linear_gaussian_data['observations_seq'],
        inference_alg_params=inference_alg_params,
        likelihood_model='multivariate_normal',
        learning_rate=1e0,
        plot_dir=inference_results_dir)

    # record elapsed time
    stop_time = timer()
    runtime = stop_time - start_time

    # record scores
    # scores, pred_cluster_labels = utils.metrics.score_predicted_clusters(
    #     true_cluster_labels=sampled_linear_gaussian_data['assigned_table_seq'],
    #     table_assignment_posteriors=inference_alg_results['table_assignment_posteriors'])

    # count number of indicators
    num_indicators = np.sum(
        np.sum(inference_alg_results['dish_eating_posteriors'], axis=0) > 0.)

    data_to_store = dict(
        inference_alg_str=inference_alg_str,
        inference_alg_params=inference_alg_params,
        inference_alg_results=inference_alg_results,
        num_indicators=num_indicators,
        # scores=scores,
        runtime=runtime)

    joblib.dump(data_to_store,
                filename=inference_results_path)

    # read results from disk
    stored_data = joblib.load(inference_results_path)

    plot_linear_gaussian.plot_inference_results(
        sampled_linear_gaussian_data=sampled_linear_gaussian_data,
        inference_alg_results=stored_data['inference_alg_results'],
        inference_alg_str=stored_data['inference_alg_str'],
        inference_alg_params=stored_data['inference_alg_params'],
        plot_dir=inference_results_dir)


def setup(args: argparse.Namespace):
    """ Create necessary directories, set seeds and load linear-Gaussian data."""

    # load Mixture of Gaussian data
    sampled_linear_gaussian_data = joblib.load(
        os.path.join(args.run_one_results_dir, 'data.joblib'))

    if args.inference_alg_str == 'R-IBP':
        inference_results_dir = f'{args.inference_alg_str}_a={args.alpha}_b={args.beta}'
        inference_alg_params = dict(
            alpha=args.alpha,
            beta=args.beta)
    else:
        raise NotImplementedError

    inference_results_dir = os.path.join(
        args.run_one_results_dir,
        inference_results_dir)
    os.makedirs(inference_results_dir, exist_ok=True)

    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    setup_results = dict(
        inference_alg_str=args.inference_alg_str,
        inference_alg_params=inference_alg_params,
        sampled_linear_gaussian_data=sampled_linear_gaussian_data,
        inference_results_dir=inference_results_dir,
    )

    return setup_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run_one_results_dir', type=str,
                        help='Path to write plots and other results to.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Pseudo-random seed for NumPy/PyTorch.')
    parser.add_argument('--inference_alg_str', type=str,
                        choices=utils.inference.inference_alg_strs,
                        help='Inference algorithm to run on dataset')
    parser.add_argument('--alpha', type=float,
                        help='IBP alpha parameter.')
    parser.add_argument('--beta', type=float,
                        help='IBP beta parameter.')
    args = parser.parse_args()
    logging.info(args)
    run_one(args)
    logging.info('Finished.')
