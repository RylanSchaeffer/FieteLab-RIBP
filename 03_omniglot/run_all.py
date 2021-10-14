"""
Launch run_one.py with each configuration of the hyperparameters.

Example usage:

03_omniglot/run_all.py
"""

import itertools
import joblib
import logging
import numpy as np
import os
import subprocess

import plot_omniglot
import utils.real


def run_all():
    # create directory
    exp_dir_path = '03_omniglot'
    results_dir_path = os.path.join(exp_dir_path, 'results')
    os.makedirs(results_dir_path, exist_ok=True)

    alphas = np.round(np.arange(1., 5.01, 1.))
    betas = np.round(np.arange(1., 5.01, 1.))
    sigma_xs = np.round(np.logspace(-2, 2, 5))
    inference_alg_strs = [
        'R-IBP',
    ]
    hyperparams = [
        alphas,
        betas,
        sigma_xs,
        inference_alg_strs,
    ]

    for alpha, beta, sigma_x, inference_alg_str in itertools.product(*hyperparams):

        run_one_results_dir_path = os.path.join(
            results_dir_path,
            f'IBP_a={alpha}_b={beta}')

        launch_run_one(
            exp_dir_path=exp_dir_path,
            run_one_results_dir_path=run_one_results_dir_path,
            inference_alg_str=inference_alg_str,
            alpha=alpha,
            beta=beta,
            sigma_x=sigma_x)

        continue


def launch_run_one(exp_dir_path: str,
                   run_one_results_dir_path: str,
                   inference_alg_str: str,
                   alpha: float,
                   beta: float,
                   sigma_x: float):

    run_one_script_path = os.path.join(exp_dir_path, 'run_one.sh')
    command_and_args = [
        'sbatch',
        run_one_script_path,
        run_one_results_dir_path,
        inference_alg_str,
        str(alpha),
        str(beta),
        str(sigma_x),
        ]

    # TODO: Figure out where the logger is logging to
    logging.info(f'Launching ' + ' '.join(command_and_args))
    subprocess.run(command_and_args)
    logging.info(f'Launched ' + ' '.join(command_and_args))


if __name__ == '__main__':
    run_all()
    logging.info('Finished.')
