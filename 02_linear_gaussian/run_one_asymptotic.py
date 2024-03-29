"""
Example usage:

02_linear_gaussian/run_one_asymptotic.py \
 --run_one_results_dir=02_linear_gaussian/results/categorical_probs=[0.2,0.2,0.2,0.2,0.2]/dataset=1 \
 --inference_alg_str=R-IBP \
 --alpha=5.91 \
 --beta=4.3
"""

import argparse
import joblib
import logging
import numpy as np
import os
from timeit import default_timer as timer
import torch

import utils.data.synthetic
import utils.inference
import utils.metrics
import utils.run_helpers


def run_one(args: argparse.Namespace):
    setup_results = setup(args=args)

    # logging.info('Running and plotting {} on dataset {}'.format(
    #     setup_results['inference_alg_str'],
    #     args.run_one_results_dir))

    asymptotic_datapoint = run_inference_alg(
        sampled_linear_gaussian_data=setup_results['sampled_linear_gaussian_data'],
        inference_alg_str=setup_results['inference_alg_str'],
        gen_model_params=setup_results['gen_model_params'],
        inference_results_dir=setup_results['inference_results_dir'])

    # logging.info('Successfully ran {} on dataset {}'.format(
    #     setup_results['inference_alg_str'],
    #     args.run_one_results_dir))

    # return asymptotic_datapoint
    logging.info(asymptotic_datapoint)


def run_inference_alg(sampled_linear_gaussian_data,
                               inference_alg_str,
                               gen_model_params,
                               inference_results_dir,
                               train_fraction: int = 1.):
    assert 0. <= train_fraction <= 1.
    # Determine the index for train-test split
    num_obs = sampled_linear_gaussian_data['observations'].shape[0]
    train_end_idx = int(num_obs * train_fraction)
    sampled_linear_gaussian_data['train_sampled_indicators'] = sampled_linear_gaussian_data[
                                                                   'sampled_indicators'][:train_end_idx]
    sampled_linear_gaussian_data['test_sampled_indicators'] = sampled_linear_gaussian_data[
                                                                  'sampled_indicators'][train_end_idx:]
    sampled_linear_gaussian_data['train_observations'] = sampled_linear_gaussian_data[
                                                             'observations'][:train_end_idx]
    sampled_linear_gaussian_data['test_observations'] = sampled_linear_gaussian_data[
                                                            'observations'][train_end_idx:]

    inference_results_path = os.path.join(
        inference_results_dir,
        'inference_alg_results.joblib')

    # if not os.path.isfile(inference_results_path):
    if True:
        # logging.info(f'Inference results not found at: {inference_results_path}')
        # logging.info('Generating inference results...')

        # run inference algorithm
        # time using timer because https://stackoverflow.com/a/25823885/4570472
        start_time = timer()
        inference_alg_results = utils.inference.run_inference_alg(
            inference_alg_str=inference_alg_str,
            observations=sampled_linear_gaussian_data['train_observations'],
            model_str='linear_gaussian',
            gen_model_params=gen_model_params,
            plot_dir=inference_results_dir)

        # record elapsed time
        stop_time = timer()
        runtime = stop_time - start_time
        # logging.info('Generated inference results.')

        training_reconstruction_error = utils.metrics.compute_reconstruction_error_linear_gaussian(
            observations=sampled_linear_gaussian_data['train_observations'],
            dish_eating_posteriors=inference_alg_results['dish_eating_posteriors'],
            features_after_last_obs=inference_alg_results['inference_alg'].features_after_last_obs())
        # logging.info(f'Computed training reconstruction error: {training_reconstruction_error}')

        # record scores
        log_posterior_predictive_results = utils.metrics.compute_log_posterior_predictive_linear_gaussian(
            train_observations=sampled_linear_gaussian_data['train_observations'],
            test_observations=sampled_linear_gaussian_data['test_observations'],
            inference_alg=inference_alg_results['inference_alg'])
        # logging.info(f"Computed log posterior predictive: mean={log_posterior_predictive_results['mean']}")

        # count number of indicators
        num_indicators = np.sum(
            np.sum(inference_alg_results['dish_eating_posteriors'], axis=0) > 0.)

        data_to_store = dict(
            sampled_linear_gaussian_data=sampled_linear_gaussian_data,
            inference_alg_str=inference_alg_str,
            inference_alg_params=gen_model_params,
            inference_alg_results=inference_alg_results,
            num_indicators=num_indicators,
            log_posterior_predictive=dict(mean=log_posterior_predictive_results['mean'],
                                          std=log_posterior_predictive_results['std']),
            training_reconstruction_error=training_reconstruction_error,
            runtime=runtime)

        # logging.info(f'Writing inference results to disk at:'
        #              f' {inference_results_path}')

        # TODO: Investigate why HMC throws the following error
        # _pickle.PicklingError: Can't pickle <function create_linear_gaussian_model.<locals>.linear_gaussian_model at 0x2b2a3d9e7170>: it's not found as utils.numpyro_models.create_linear_gaussian_model.<locals>.linear_gaussian_model
        # Then remove this del
        # This will break regardless inside plot_linear_gaussian.plot_run_one_inference_results
        if inference_alg_str == 'HMC-Gibbs':
            del data_to_store['inference_alg_results']['inference_alg']
        joblib.dump(data_to_store,
                    filename=inference_results_path)

    # logging.info(f'Loading inference from {inference_results_path}')

    # read results from disk
    # stored_data = joblib.load(inference_results_path)

    num_inferred_features = inference_alg_results['inference_alg'].features_after_last_obs().shape[0]
    num_true_features = sampled_linear_gaussian_data['gaussian_params']['means'].shape[0]
    feature_ratio = 1. * num_inferred_features / num_true_features

    return (num_obs, feature_ratio)
    # return stored_data['inference_alg_results']['inference_alg'].features_after_last_obs().shape[0] # number of inferred features

    # plot_linear_gaussian.plot_run_one_inference_results(
    #     sampled_linear_gaussian_data=sampled_linear_gaussian_data,
    #     inference_alg_results=stored_data['inference_alg_results'],
    #     inference_alg_str=stored_data['inference_alg_str'],
    #     inference_alg_params=stored_data['inference_alg_params'],
    #     log_posterior_predictive_dict=stored_data['log_posterior_predictive'],
    #     plot_dir=inference_results_dir)
    # logging.info('Plotted inference algorithm results.')


def setup(args: argparse.Namespace):
    """ Create necessary directories, set seeds and load linear-Gaussian data."""

    inference_results_dir = f'{args.inference_alg_str}_a={args.alpha}_b={args.beta}'

    inference_results_dir = os.path.join(
        args.run_one_results_dir,
        inference_results_dir)
    os.makedirs(inference_results_dir, exist_ok=True)

    utils.run_helpers.create_logger(run_dir=inference_results_dir)

    # logging.info(args)

    # load data
    sampled_linear_gaussian_data = joblib.load(
        os.path.join(args.run_one_results_dir, 'data.joblib'))

    gen_model_params = dict(
        IBP=dict(
            alpha=args.alpha,
            beta=args.beta))

    gen_model_params['feature_prior_params'] = sampled_linear_gaussian_data[
        'feature_prior_params']
    gen_model_params['likelihood_params'] = sampled_linear_gaussian_data[
        'likelihood_params']

    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    setup_results = dict(
        inference_alg_str=args.inference_alg_str,
        gen_model_params=gen_model_params,
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
                        choices=utils.inference.inference_algs,
                        help='Inference algorithm to run on dataset')
    parser.add_argument('--alpha', type=float,
                        help='IBP alpha parameter.')
    parser.add_argument('--beta', type=float,
                        help='IBP beta parameter.')
    args = parser.parse_args()
    run_one(args)
    # logging.info('Finished.')

