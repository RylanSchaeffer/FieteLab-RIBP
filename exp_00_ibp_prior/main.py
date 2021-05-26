# TODO: switch to log colorscale scaling
import numpy as np
import os
import scipy.stats

import plot
from utils.data import vectorized_sample_sequence_from_ibp


def main():

    # set seed
    np.random.seed(1)

    # create directories
    exp_dir = 'exp_00_ibp_prior'
    plot_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    T = 50  # max time
    num_samples = 5000  # number of samples to draw from IBP(alpha)
    alphas = [1.1, 10.37, 15.78, 30.91]  # IBP parameter
    # alphas = [30.91]  # IBP parameter

    sampled_customers_dishes_by_alpha = sample_from_ibp(
        T=T,
        alphas=alphas,
        exp_dir=exp_dir,
        num_samples=num_samples)

    analytical_dish_distribution_poisson_rate_by_alpha_by_T = construct_analytical_indian_dish_distribution(
        T=T,
        alphas=alphas)

    analytical_customers_dishes_by_alpha = construct_analytical_ibp(
        T=T,
        alphas=alphas,
        exp_dir=exp_dir)

    plot.plot_indian_buffet_num_dishes_dist_by_customer_num(
        analytical_dish_distribution_poisson_rate_by_alpha_by_T=analytical_dish_distribution_poisson_rate_by_alpha_by_T,
        plot_dir=plot_dir)

    plot.plot_recursion_visualization(
        analytical_customers_dishes_by_alpha=analytical_customers_dishes_by_alpha,
        analytical_dish_distribution_poisson_rate_by_alpha_by_T=analytical_dish_distribution_poisson_rate_by_alpha_by_T,
        plot_dir=plot_dir)

    plot.plot_analytics_vs_monte_carlo_customer_dishes(
        sampled_customers_dishes_by_alpha=sampled_customers_dishes_by_alpha,
        analytical_customers_dishes_by_alpha=analytical_customers_dishes_by_alpha,
        plot_dir=plot_dir)

    num_reps = 10
    error_means_per_num_samples_per_alpha, error_sems_per_num_samples_per_alpha = \
        calc_analytical_vs_monte_carlo_mse(
            T=T,
            alphas=alphas,
            exp_dir=exp_dir,
            num_reps=num_reps,
            num_samples=num_samples,
            analytical_customer_dishes_by_alpha=analytical_customers_dishes_by_alpha)

    plot.plot_analytical_vs_monte_carlo_mse(
        error_means_per_num_samples_per_alpha=error_means_per_num_samples_per_alpha,
        error_sems_per_num_samples_per_alpha=error_sems_per_num_samples_per_alpha,
        num_reps=num_reps,
        plot_dir=plot_dir)


def calc_analytical_vs_monte_carlo_mse(T: int,
                                       alphas,
                                       exp_dir,
                                       num_reps: int,
                                       num_samples: int,
                                       analytical_customer_dishes_by_alpha):

    sample_subset_sizes = np.logspace(1, np.log10(num_samples), 5).astype(np.int)

    rep_errors = np.zeros(shape=(num_reps, len(alphas), len(sample_subset_sizes)))

    for rep_idx in range(num_reps):
        # draw sample from CRP
        sampled_customer_dishes_by_alpha = sample_from_ibp(
            T=T,
            alphas=alphas,
            exp_dir=exp_dir,
            num_samples=num_samples,
            rep_idx=rep_idx)

        for alpha_idx, alpha in enumerate(alphas):
            # for each subset of data, calculate the error
            for sample_idx, num_samples in enumerate(sample_subset_sizes):
                diff = np.subtract(
                        np.mean(sampled_customer_dishes_by_alpha[alpha][:num_samples],
                                axis=0),
                        analytical_customer_dishes_by_alpha[alpha])
                rep_error = np.square(np.linalg.norm(diff[~np.isnan(diff)]))
                rep_errors[rep_idx, alpha_idx, sample_idx] = rep_error

    mean_errors_per_num_samples_per_alpha, sem_errors_per_num_samples_per_alpha = {}, {}
    for alpha_idx, alpha in enumerate(alphas):
        mean_errors_per_num_samples_per_alpha[alpha] = {
            num_sample: error for num_sample, error in
            zip(sample_subset_sizes, np.mean(rep_errors[:, alpha_idx, :], axis=0))}
        sem_errors_per_num_samples_per_alpha[alpha] = {
            num_sample: error for num_sample, error in
            zip(sample_subset_sizes, scipy.stats.sem(rep_errors[:, alpha_idx, :], axis=0))}

    return mean_errors_per_num_samples_per_alpha, sem_errors_per_num_samples_per_alpha


def construct_analytical_customers_dishes(T, alpha):

    # shape: (number of customers, number of dishes)
    assert alpha > 0
    alpha = float(alpha)

    # heuristic: 3 * expected number
    # needs to match whatever the sampled IBP max is, otherwise shapes disagree
    max_dishes = int(3 * alpha * np.sum(1 / (1 + np.arange(T))))
    analytical_customers_dishes = np.zeros(shape=(T + 1, max_dishes + 1))
    analytical_customer_dishes_running_sum = np.zeros(shape=(T + 1, max_dishes + 1))
    dish_indices = np.arange(max_dishes + 1)

    # customer 1 samples only new dishes
    new_dishes_rate = alpha / 1
    analytical_customers_dishes[1, :] = np.cumsum(scipy.stats.poisson.pmf(dish_indices[::-1], mu=new_dishes_rate))[
                                        ::-1]
    analytical_customer_dishes_running_sum[1, :] = analytical_customers_dishes[1, :]
    total_dishes_rate_running_sum = new_dishes_rate

    # all subsequent customers sample new dishes
    for customer_num in range(2, T + 1):
        analytical_customers_dishes[customer_num, :] = analytical_customer_dishes_running_sum[customer_num - 1,
                                                       :] / customer_num
        new_dishes_rate = alpha / customer_num
        cdf_lambda_t_minus_1 = scipy.stats.poisson.cdf(dish_indices, mu=total_dishes_rate_running_sum)
        cdf_lambda_t = scipy.stats.poisson.cdf(dish_indices, mu=total_dishes_rate_running_sum + new_dishes_rate)
        cdf_diff = np.subtract(cdf_lambda_t_minus_1, cdf_lambda_t)
        analytical_customers_dishes[customer_num, :] += cdf_diff
        analytical_customer_dishes_running_sum[customer_num, :] = \
            np.add(analytical_customer_dishes_running_sum[customer_num - 1, :],
                   analytical_customers_dishes[customer_num, :])
        total_dishes_rate_running_sum += new_dishes_rate

    return analytical_customers_dishes[1:, 1:]


def construct_analytical_ibp(T, alphas, exp_dir):
    analytical_customers_dishes_by_alpha = {}
    for alpha in alphas:
        ibp_analytics_path = os.path.join(exp_dir, f'ibp_analytics_{alpha}.npz')
        if os.path.isfile(ibp_analytics_path):
            npz_file = np.load(ibp_analytics_path)
            analytical_customers_dishes = npz_file['analytical_customers_dishes']
            assert analytical_customers_dishes.shape[0] == T
        else:
            analytical_customers_dishes = construct_analytical_customers_dishes(
                T=T,
                alpha=alpha)
            np.savez(analytical_customers_dishes=analytical_customers_dishes,
                     file=ibp_analytics_path)
        analytical_customers_dishes_by_alpha[alpha] = analytical_customers_dishes
    return analytical_customers_dishes_by_alpha


def construct_analytical_indian_dish_distribution(T, alphas):

    num_customers = 1 + np.arange(T)
    dish_distribution_poisson_rate_by_alpha_by_T = {}
    for alpha in alphas:
        dish_distribution_poisson_rate_by_alpha_by_T[alpha] = {}
        poisson_rate = 0
        for t in num_customers:
            poisson_rate += alpha / t
            dish_distribution_poisson_rate_by_alpha_by_T[alpha][t] = poisson_rate
    return dish_distribution_poisson_rate_by_alpha_by_T


def sample_from_ibp(T,
                    alphas,
                    exp_dir,
                    num_samples,
                    rep_idx=0):

    # generate Monte Carlo samples from IBP(alpha)
    sampled_customer_dishes_by_alpha = {}
    for alpha in alphas:
        ibp_samples_path = os.path.join(exp_dir, f'ibp_sample_{alpha}_rep_idx={rep_idx}.npz')
        if os.path.isfile(ibp_samples_path):
            ibp_sample_data = np.load(ibp_samples_path)
            sampled_customer_dishes = ibp_sample_data['sampled_customer_dishes']
            assert sampled_customer_dishes.shape[0] == num_samples
            assert sampled_customer_dishes.shape[1] == T
            print(f'Loaded samples for {ibp_samples_path}')
        else:
            sampled_customer_dishes = vectorized_sample_sequence_from_ibp(
                T=np.full(shape=num_samples, fill_value=T),
                alpha=np.full(shape=num_samples, fill_value=alpha))
            sampled_customer_dishes = np.stack(sampled_customer_dishes)
            np.savez(file=ibp_samples_path,
                     sampled_customer_dishes=sampled_customer_dishes)
            print(f'Generated samples for {ibp_samples_path}')
        sampled_customer_dishes_by_alpha[alpha] = sampled_customer_dishes

    return sampled_customer_dishes_by_alpha


if __name__ == '__main__':
    main()
