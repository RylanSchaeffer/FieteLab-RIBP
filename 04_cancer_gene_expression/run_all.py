"""
Launch run_one.py with each configuration of the hyperparameters.

Example usage:

04_cancer_gene_expression/run_all.py
"""

import itertools
import joblib
import logging
import numpy as np
import os
import subprocess

import plot_cancer_gene_expression


def run_all():
    # create directory
    exp_dir_path = '04_cancer_gene_expression'
    results_dir_path = os.path.join(exp_dir_path, 'results')
    os.makedirs(results_dir_path, exist_ok=True)

    inference_alg_strs = [
        'R-IBP',
    ]
    alphas = [1., 2.5, 5., 7.5]
    betas = [1., 2.5, 5., 7.5]
    feature_prior_cov_scalings = [0.1, 1., 5., 10.]
    sigma_xs = [0.1, 1., 5., 10.]
    seeds = list(range(3))

    hyperparams = [
        inference_alg_strs,
        alphas,
        betas,
        feature_prior_cov_scalings,
        sigma_xs,
        seeds]

    # independently launch inference
    for inference_alg_str, alpha, beta, feature_prior_cov_scaling, \
        sigma_x, seed in itertools.product(*hyperparams):

        run_str = f'{inference_alg_str}_a={alpha}_b={beta}_' \
                  f'featurecov={feature_prior_cov_scaling}_sigmax={sigma_x}_' \
                  f'seed={seed}'

        run_one_results_dir_path = os.path.join(
            results_dir_path,
            run_str)

        launch_run_one(
            exp_dir_path=exp_dir_path,
            run_one_results_dir_path=run_one_results_dir_path,
            inference_alg_str=inference_alg_str,
            alpha=alpha,
            beta=beta,
            feature_prior_cov_scaling=feature_prior_cov_scaling,
            sigma_x=sigma_x,
            seed=seed)


def launch_run_one(exp_dir_path: str,
                   run_one_results_dir_path: str,
                   inference_alg_str: str,
                   alpha: float,
                   beta: float,
                   feature_prior_cov_scaling: float,
                   sigma_x: float,
                   seed: int):

    run_one_script_path = os.path.join(exp_dir_path, 'run_one.sh')
    command_and_args = [
        'sbatch',
        run_one_script_path,
        run_one_results_dir_path,
        inference_alg_str,
        str(alpha),
        str(beta),
        str(feature_prior_cov_scaling),
        str(sigma_x),
        str(seed)]

    # TODO: Figure out where the logger is logging to
    logging.info(f'Launching ' + ' '.join(command_and_args))
    subprocess.run(command_and_args)
    logging.info(f'Launched ' + ' '.join(command_and_args))


if __name__ == '__main__':
    run_all()
    logging.info('Finished.')
