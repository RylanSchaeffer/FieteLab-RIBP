"""
Perform inference in a linear-Gaussian model X = ZA + noise
for the specified inference algorithm.

Example usage:

02_linear_gaussian/run_one.py \
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
import pandas as pd

import utils.data.synthetic
import utils.inference
import utils.metrics
import utils.run_helpers
from typing import Union


def run_one(args: argparse.Namespace):
    setup_results = setup(args=args)

    # logging.info('Running and plotting {} on dataset {}'.format(
    #     setup_results['inference_alg_str'],
    #     args.run_one_results_dir))

    melted_data_path = run_and_plot_inference_alg(
        sampled_linear_gaussian_data=setup_results['sampled_linear_gaussian_data'],
        inference_alg_str=setup_results['inference_alg_str'],
        gen_model_params=setup_results['gen_model_params'],
        inference_results_dir=setup_results['inference_results_dir'],
        data_dim=setup_results['data_dim'])

    # logging.info('Successfully ran and plotted {} on dataset {}'.format(
    #     setup_results['inference_alg_str'],
    #     args.run_one_results_dir))
    logging.info(melted_data_path)


def run_and_plot_inference_alg(sampled_linear_gaussian_data,
                               inference_alg_str,
                               gen_model_params,
                               inference_results_dir,
                               train_fraction: int = .80,
                               data_dim: int=2,):
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
        f'fig6_st1_run_one_inference_alg_results.joblib')

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
    stored_data = joblib.load(inference_results_path)

    # TODO: separate train and test data when plotting, otherwise arrays of unequal length
    # logging.info('Plotting inference algorithm results...')
    # plot_linear_gaussian.plot_run_one_inference_results(
    #     sampled_linear_gaussian_data=sampled_linear_gaussian_data,
    #     inference_alg_results=,
    #     inference_alg_str=stored_data['inference_alg_str'],
    #     inference_alg_params=stored_data['inference_alg_params'],
    #     log_posterior_predictive_dict=stored_data['log_posterior_predictive'],
    #     plot_dir=inference_results_dir)
    # logging.info('Plotted inference algorithm results.')
    inference_alg_results = stored_data['inference_alg_results']
    return run_one_save_num_features_by_num_obs_using_poisson_rates(
        num_dishes_poisson_rate_priors=inference_alg_results['num_dishes_poisson_rate_priors'],
        num_dishes_poisson_rate_posteriors=inference_alg_results['num_dishes_poisson_rate_posteriors'],
        indicators=sampled_linear_gaussian_data['train_sampled_indicators'],
        save_dir=inference_results_dir,
        data_dim=data_dim)



def run_one_save_num_features_by_num_obs_using_poisson_rates(indicators: Union[np.ndarray, None],
                                                            num_dishes_poisson_rate_priors,
                                                            num_dishes_poisson_rate_posteriors,
                                                            save_dir,
                                                            data_dim):
    seq_length = num_dishes_poisson_rate_priors.shape[0]
    obs_indices = 1 + np.arange(seq_length)  # remember, we started with t = 0
    if indicators is None:
        real_num_dishes = np.full_like(
            num_dishes_poisson_rate_priors[:, 0],
            fill_value=np.nan, )
    else:
        real_num_dishes = np.concatenate(
            [np.sum(np.minimum(np.cumsum(indicators, axis=0), 1), axis=1)])

    # r'$q(\Lambda_t|o_{< t})$'
    # r'$q(\Lambda_t|o_{\leq t})$'
    data_to_plot = pd.DataFrame.from_dict({
        'obs_idx': obs_indices,
        'data_dim': np.array([data_dim]*seq_length),
        'dish_ratio': real_num_dishes / num_dishes_poisson_rate_posteriors[:, 0],
        # 'Prior': num_dishes_poisson_rate_priors[:, 0],
        # 'Posterior': num_dishes_poisson_rate_posteriors[:, 0],
    })

    if indicators is None:
        data_to_plot.drop(axis='columns', labels='True', inplace=True)

    # melted_data_to_plot = data_to_plot.melt(
    #     id_vars=['obs_idx'],  # columns to keep
    #     var_name='quantity',  # new column name for previous columns' headers
    #     value_name='num_dishes',  # new column name for values
    # )
    # print("MELTED DATA TO PLOT:", melted_data_to_plot)
    data_path = os.path.join(save_dir,
                             f'data_to_plot.pkl')
    data_to_plot.to_pickle(data_path)
    return str(data_path)


def setup(args: argparse.Namespace):
    """ Create necessary directories, set seeds and load linear-Gaussian data."""

    inference_results_dir = f'{args.inference_alg_str}_a={args.alpha}_b={args.beta}_datadim={args.data_dim}'

    inference_results_dir = os.path.join(
        args.run_one_results_dir,
        inference_results_dir)
    os.makedirs(inference_results_dir, exist_ok=True)

    utils.run_helpers.create_logger(run_dir=inference_results_dir)

    # logging.info(args)

    # load data
    sampled_linear_gaussian_data = joblib.load(
        os.path.join(args.run_one_results_dir, f'data.joblib'))

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
        data_dim=args.data_dim,
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
    parser.add_argument('--data_dim', type=int,
                        help='Data dimension parameter.')
    args = parser.parse_args()
    run_one(args)
    # logging.info('Finished.')
