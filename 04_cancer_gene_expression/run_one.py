"""
Perform inference in a linear-Gaussian model X = ZA + noise
for the specified inference algorithm on the 2016 cancer
gene expression dataset.

Example usage:

04_cancer_gene_expression/run_one.py \
 --run_one_results_dir=04_cancer_gene_expression/results/tmp/ \
 --inference_alg_str=R-IBP \
 --alpha=5.91 \
 --beta=4.3 \
 --sigma_x=2. \
 --feature_prior_cov_scaling=10. \
 --scale_prior_cov_scaling=1.
"""

import argparse
import joblib
import logging
import numpy as np
import os
from timeit import default_timer as timer
import torch

import plot_cancer_gene_expression
from utils.data.real import load_dataset_cancer_gene_expression_2016
import utils.inference
import utils.metrics
import utils.run_helpers


def run_one(args: argparse.Namespace):

    setup_results = setup(args=args)

    print('Running and plotting {} on dataset {}'.format(
        setup_results['inference_alg_str'],
        args.run_one_results_dir))

    run_and_plot_inference_alg(
        cancer_gene_expression_data=setup_results['cancer_gene_expression_data'],
        inference_alg_str=setup_results['inference_alg_str'],
        gen_model_params=setup_results['gen_model_params'],
        inference_results_dir=setup_results['inference_results_dir'])

    print('Successfully ran and plotted {} on dataset {}'.format(
        setup_results['inference_alg_str'],
        args.run_one_results_dir))


def run_and_plot_inference_alg(cancer_gene_expression_data,
                               inference_alg_str,
                               gen_model_params,
                               inference_results_dir,
                               train_fraction: int = .80):

    assert 0. <= train_fraction <= 1.

    # Determine the index for train-test split
    num_obs = cancer_gene_expression_data['observations'].shape[0]
    train_end_idx = int(num_obs * train_fraction)
    cancer_gene_expression_data['train_observations'] = \
        cancer_gene_expression_data['observations'][:train_end_idx]
    cancer_gene_expression_data['test_observations'] = \
        cancer_gene_expression_data['observations'][train_end_idx:]

    inference_results_path = os.path.join(
        inference_results_dir,
        'inference_alg_results.joblib')

    if not os.path.isfile(inference_results_path):

        print(f'Inference results not found at: {inference_results_path}')
        print('Generating inference results...')

        # run inference algorithm
        # time using timer because https://stackoverflow.com/a/25823885/4570472
        start_time = timer()

        inference_alg_results = utils.inference.run_inference_alg(
            inference_alg_str=inference_alg_str,
            observations=cancer_gene_expression_data['train_observations'],
            model_str='linear_gaussian',
            gen_model_params=gen_model_params,
            plot_dir=inference_results_dir)

        # record elapsed time
        stop_time = timer()
        runtime = stop_time - start_time
        print('Generated inference results.')

        # record scores
        log_posterior_predictive_results = utils.metrics.compute_log_posterior_predictive_factor_analysis(
            train_observations=cancer_gene_expression_data['train_observations'],
            test_observations=cancer_gene_expression_data['test_observations'],
            inference_alg=inference_alg_results['inference_alg'])
        print('Computed log posterior predictive.')

        # count number of indicators
        num_indicators = np.sum(
            np.sum(inference_alg_results['dish_eating_posteriors'], axis=0) > 0.)

        data_to_store = dict(
            inference_alg_str=inference_alg_str,
            inference_alg_params=gen_model_params,
            inference_alg_results=inference_alg_results,
            num_indicators=num_indicators,
            log_posterior_predictive=dict(mean=log_posterior_predictive_results['mean'],
                                          std=log_posterior_predictive_results['std']),
            runtime=runtime)

        print(f'Writing inference results to disk at:'
                     f' {inference_results_path}')
        joblib.dump(data_to_store,
                    filename=inference_results_path)

    print(f'Loading inference from {inference_results_path}')

    # read results from disk
    stored_data = joblib.load(inference_results_path)

    print('Plotting inference algorithm results...')
    # plot_cancer_gene_expression.plot_run_one_inference_results(
    #     sampled_cancer_gene_expression_data=sampled_cancer_gene_expression_data,
    #     inference_alg_results=stored_data['inference_alg_results'],
    #     inference_alg_str=stored_data['inference_alg_str'],
    #     inference_alg_params=stored_data['inference_alg_params'],
    #     log_posterior_predictive_dict=stored_data['log_posterior_predictive'],
    #     plot_dir=inference_results_dir)
    print('Plotted inference algorithm results.')


def setup(args: argparse.Namespace):
    """ Create necessary directories, set seeds and load linear-Gaussian data."""

    inference_results_dir = args.run_one_results_dir
    print(f'Inference results dir: {inference_results_dir}')
    os.makedirs(inference_results_dir, exist_ok=True)

    utils.run_helpers.create_logger(run_dir=inference_results_dir)

    print(args)

    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Loading 2016 Cancer Gene Expression...')
    cancer_gene_expression_data = load_dataset_cancer_gene_expression_2016()
    print('Loaded 2016 Cancer Gene Expression!')

    # Permute the order of the data
    n_obs = cancer_gene_expression_data['observations'].shape[0]
    permutation = np.random.permutation(np.arange(n_obs))
    cancer_gene_expression_data['observations'] = \
        cancer_gene_expression_data['observations'][permutation]
    cancer_gene_expression_data['labels'] = \
        cancer_gene_expression_data['labels'][permutation]
    print('Randomly permuted the order of 2016 Cancer Gene Expression data.')

    from sklearn.random_projection import GaussianRandomProjection
    jl_projection = GaussianRandomProjection(n_components='auto',
                                             eps=args.jl_eps)
    projected_observations = jl_projection.fit_transform(
        cancer_gene_expression_data['observations'])
    print(f'Randomly projected 2016 Cancer Gene Expression data to {projected_observations.shape[1]} dimensions.')
    cancer_gene_expression_data['observations'] = projected_observations

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaled_projected_observations = scaler.fit_transform(projected_observations)
    # cancer_gene_expression_data['observations'] = \
    #     scaled_projected_observations
    # print('Rescaled the 2016 Cancer Gene Expression data.')

    gen_model_params = dict(
        IBP=dict(
            alpha=args.alpha,
            beta=args.beta),
        feature_prior_params=dict(
            feature_prior_cov_scaling=args.feature_prior_cov_scaling,
        ),
        likelihood_params=dict(
            sigma_x=args.sigma_x),
        )

    setup_results = dict(
        inference_alg_str=args.inference_alg_str,
        gen_model_params=gen_model_params,
        cancer_gene_expression_data=cancer_gene_expression_data,
        inference_results_dir=inference_results_dir,
    )

    return setup_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run_one_results_dir', type=str,
                        help='Path to write plots and other results to.')
    parser.add_argument('--seed', type=int,
                        help='Pseudo-random seed for NumPy/PyTorch.')
    parser.add_argument('--inference_alg_str', type=str,
                        choices=utils.inference.inference_algs,
                        help='Inference algorithm to run on dataset')
    parser.add_argument('--alpha', type=float,
                        help='IBP alpha parameter.')
    parser.add_argument('--beta', type=float,
                        help='IBP beta parameter.')
    parser.add_argument('--sigma_x', type=float,
                        help='Likelihood (noise) covariance parameter.')
    parser.add_argument('--feature_prior_cov_scaling', type=float,
                        help='Scale on feature A_k prior covariance.')
    parser.add_argument('--jl_eps', type=float,
                        default=0.25,
                        help='Epsilon for JL projection.')
    args = parser.parse_args()
    run_one(args)
    print('Finished.')
