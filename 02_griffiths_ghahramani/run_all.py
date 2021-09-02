"""
Launch run_one.py with each configuration of the hyperparameters.

Example usage:

02_griffiths_ghahramani/run_all.py
"""

import itertools
import joblib
import logging
import numpy as np
import os
import subprocess

import plot_linear_gaussian
import utils.data


def run_all():
    # create directory
    exp_dir_path = '02_griffiths_ghahramani'
    results_dir_path = os.path.join(exp_dir_path, 'results')
    os.makedirs(results_dir_path, exist_ok=True)

    feature_samplings = [
        ('categorical', dict(probs=np.array([0.5, 0.5, 0.5, 0.5]))),
    ]

    num_datasets = 10
    num_customers = 100

    alphas = np.round(np.linspace(1.1, 5.91, 20), 2)
    betas = np.round(np.linspace(0.3, 8.7, 20), 2)
    inference_alg_strs = ['R-IBP']
    hyperparams = [alphas, betas, inference_alg_strs]

    # generate several datasets and independently launch inference
    for (indicator_sampling, indicator_sampling_params), dataset_idx in \
            itertools.product(feature_samplings, range(num_datasets)):

        logging.info(f'Sampling: {indicator_sampling}, Dataset Index: {dataset_idx}')
        sampled_linear_gaussian_data = utils.data.sample_from_griffiths_ghahramani_2005(
            num_obs=num_customers,
            indicator_sampling_params=indicator_sampling_params)

        # save dataset
        run_one_results_dir_path = os.path.join(
            results_dir_path,
            sampled_linear_gaussian_data['indicator_sampling_descr_str'],
            f'dataset={dataset_idx}')
        os.makedirs(run_one_results_dir_path, exist_ok=True)
        joblib.dump(sampled_linear_gaussian_data,
                    filename=os.path.join(run_one_results_dir_path, 'data.joblib'))

        plot_linear_gaussian.plot_sample_from_linear_gaussian(
            features=sampled_linear_gaussian_data['features'],
            observations_seq=sampled_linear_gaussian_data['observations_seq'],
            plot_dir=run_one_results_dir_path)

        for alpha, beta, inference_alg_str in itertools.product(*hyperparams):
            launch_run_one(
                exp_dir_path=exp_dir_path,
                run_one_results_dir_path=run_one_results_dir_path,
                inference_alg_str=inference_alg_str,
                alpha=alpha,
                beta=beta)
            # continue


def launch_run_one(exp_dir_path: str,
                   run_one_results_dir_path: str,
                   inference_alg_str: str,
                   alpha: float,
                   beta: float):

    run_one_script_path = os.path.join(exp_dir_path, 'run_one.sh')
    command_and_args = [
        'sbatch',
        run_one_script_path,
        run_one_results_dir_path,
        inference_alg_str,
        str(alpha),
        str(beta)]

    # TODO: Figure out where the logger is logging to
    logging.info(f'Launching ' + ' '.join(command_and_args))
    subprocess.run(command_and_args)
    logging.info(f'Launched ' + ' '.join(command_and_args))


if __name__ == '__main__':
    run_all()
    logging.info('Finished.')
