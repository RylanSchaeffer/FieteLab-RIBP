"""
Launch run_one.py with each configuration of the parameters, to compare the
analytical DCRP marginal distribution against Monte-Carlo estimates of the DCRP
marginal distribution.

Example usage:

00_prior/run_all.py
"""

import itertools
import logging
import os
import subprocess

import utils.helpers


def run_all():
    # create directory
    exp_dir_path = '00_prior'
    results_dir_path = os.path.join(exp_dir_path, 'results')
    os.makedirs(results_dir_path, exist_ok=True)

    num_customers = [50]
    num_mc_samples = [5000]  # number of Monte Carlo samples to draw
    alphas = [1.1, 10.78, 15.37, 30.91]
    betas = [0.3, 5.6, 12.9, 21.3]

    hyperparams = [num_customers, num_mc_samples, alphas, betas]
    for num_customer, num_mc_sample, alpha, beta in itertools.product(*hyperparams):
        launch_run_one(
            exp_dir_path=exp_dir_path,
            results_dir_path=results_dir_path,
            num_customer=num_customer,
            num_mc_sample=num_mc_sample,
            alpha=alpha,
            beta=beta)

    # TODO: re-add this functionality
    #     num_reps = 10
    #     error_means_per_num_samples_per_alpha, error_sems_per_num_samples_per_alpha = \
    #         calc_analytical_vs_monte_carlo_mse(
    #             T=T,
    #             alphas=alphas,
    #             exp_dir=exp_dir,
    #             num_reps=num_reps,
    #             sample_subset_size=num_samples,
    #             analytical_customer_tables_by_alpha=analytical_customer_tables_by_alpha)
    #
    #     plot.plot_analytical_vs_monte_carlo_mse(
    #         error_means_per_num_samples_per_alpha=error_means_per_num_samples_per_alpha,
    #         error_sems_per_num_samples_per_alpha=error_sems_per_num_samples_per_alpha,
    #         num_reps=num_reps,
    #         plot_dir=plot_dir)

    # def calc_analytical_vs_monte_carlo_mse(T: int,
    #                                        alphas,
    #                                        exp_dir,
    #                                        num_reps: int,
    #                                        sample_subset_size: int,
    #                                        analytical_customer_tables_by_alpha):
    #
    #     sample_subset_sizes = np.logspace(1, 4, 5).astype(np.int)
    #
    #     rep_errors = np.zeros(shape=(num_reps, len(alphas), len(sample_subset_sizes)))
    #
    #     for rep_idx in range(num_reps):
    #         # draw sample from CRP
    #         _, sampled_customer_tables_by_alpha = sample_from_dcrp(
    #             T=T,
    #             alphas=alphas,
    #             exp_dir=exp_dir,
    #             num_samples=sample_subset_size,
    #             rep_idx=rep_idx)
    #
    #         for alpha_idx, alpha in enumerate(alphas):
    #             # for each subset of data, calculate the error
    #             for sample_idx, sample_subset_size in enumerate(sample_subset_sizes):
    #                 rep_error = np.square(np.linalg.norm(
    #                     np.subtract(
    #                         np.mean(sampled_customer_tables_by_alpha[alpha][:sample_subset_size],
    #                                 axis=0),
    #                         analytical_customer_tables_by_alpha[alpha])
    #                 ))
    #                 rep_errors[rep_idx, alpha_idx, sample_idx] = rep_error
    #
    #     means_per_num_samples_per_alpha, sems_per_num_samples_per_alpha = {}, {}
    #     for alpha_idx, alpha in enumerate(alphas):
    #         means_per_num_samples_per_alpha[alpha] = {
    #             num_sample: error for num_sample, error in
    #             zip(sample_subset_sizes, np.mean(rep_errors[:, alpha_idx, :], axis=0))}
    #         sems_per_num_samples_per_alpha[alpha] = {
    #             num_sample: error for num_sample, error in
    #             zip(sample_subset_sizes, scipy.stats.sem(rep_errors[:, alpha_idx, :], axis=0))}
    #
    #     return means_per_num_samples_per_alpha, sems_per_num_samples_per_alpha


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
