"""
Launch run_one.py with each configuration of the params, to compare the
analytical RIBP marginal distribution against Monte-Carlo estimates of the RIBP
marginal distribution.

Example usage:

01_prior/run_all.py
"""

import itertools
import logging
import os
import subprocess


def run_all():
    # create directory
    exp_dir_path = '01_prior'
    results_dir_path = os.path.join(exp_dir_path, 'results')
    os.makedirs(results_dir_path, exist_ok=True)

    num_customers = [50]
    num_mc_samples = [5000]  # number of Monte Carlo samples to draw
    alphas = [1.1, 10.78, 15.37]
    betas = [2.3, 5.6, 12.9]

    hyperparams = [num_customers, num_mc_samples, alphas, betas]
    for num_customer, num_mc_sample, alpha, beta in itertools.product(*hyperparams):
        launch_run_one(
            exp_dir_path=exp_dir_path,
            results_dir_path=results_dir_path,
            num_customer=num_customer,
            num_mc_sample=num_mc_sample,
            alpha=alpha,
            beta=beta)


def launch_run_one(exp_dir_path: str,
                   results_dir_path: str,
                   num_customer: int,
                   num_mc_sample: int,
                   alpha: float,
                   beta: float):

    run_one_script_path = os.path.join(exp_dir_path, 'run_one.sh')
    command_and_args = [
        'sbatch',
        run_one_script_path,
        results_dir_path,
        str(num_customer),
        str(num_mc_sample),
        str(alpha),
        str(beta)]

    # TODO: Figure out where the logger is logging to
    logging.info(f'Launching ' + ' '.join(command_and_args))
    subprocess.run(command_and_args)
    logging.info(f'Launched ' + ' '.join(command_and_args))


if __name__ == '__main__':
    run_all()
    logging.info('Finished.')
