"""
Example usage:

02_linear_gaussian/analyze_asymptotics.py
"""

import itertools
import joblib
import logging
import numpy as np
import os
import subprocess

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats
import torch
import torchvision
from typing import Dict, Union

import utils.data.synthetic

def sample_ibp(num_mc_sample: int,
               num_customer: int,
               num_true_features: int,
               alpha: float,
               beta: float) -> Dict[str, np.ndarray]:
    assert alpha > 0.
    assert beta > 0.

    # preallocate results
    # use 10 * expected number of dishes as heuristic
    max_dishes = max(10 * int(alpha * beta * np.sum(1 / (1 + np.arange(num_customer)))), num_true_features)
    sampled_dishes_by_customer_idx = np.zeros(
        shape=(num_mc_sample, num_customer, max_dishes),
        dtype=np.int16)
    cum_sampled_dishes_by_customer_idx = np.zeros(
        shape=(num_mc_sample, num_customer, max_dishes),
        dtype=np.int16)
    num_dishes_by_customer_idx = np.zeros(
        shape=(num_mc_sample, num_customer, max_dishes),
        dtype=np.int16)

    for smpl_idx in range(num_mc_sample):
        current_num_dishes = 0
        for cstmr_idx in range(1, num_customer + 1):
            # sample existing dishes
            prob_new_customer_sampling_dish = cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 2, :] / \
                                              (beta + cstmr_idx - 1)
            existing_dishes_for_new_customer = np.random.binomial(
                n=1,
                p=prob_new_customer_sampling_dish[np.newaxis, :])[0]
            sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1,
            :] = existing_dishes_for_new_customer  # .astype(np.int)

            # sample number of new dishes for new customer
            # subtract 1 from to cstmr_idx because of 1-based iterating
            num_new_dishes = np.random.poisson(alpha * beta / (beta + cstmr_idx - 1))
            start_dish_idx = current_num_dishes
            end_dish_idx = current_num_dishes + num_new_dishes
            sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, start_dish_idx:end_dish_idx] = 1

            # increment current num dishes
            current_num_dishes += num_new_dishes
            num_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, current_num_dishes] = 1

            # increment running sums
            cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, :] = np.add(
                cum_sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 2, :],
                sampled_dishes_by_customer_idx[smpl_idx, cstmr_idx - 1, :])

    sample_ibp_results = {
        'cum_sampled_dishes_by_customer_idx': cum_sampled_dishes_by_customer_idx,
        'sampled_dishes_by_customer_idx': sampled_dishes_by_customer_idx,
        'num_dishes_by_customer_idx': num_dishes_by_customer_idx,
    }

    # import matplotlib.pyplot as plt
    # mc_avg = np.mean(sample_ibp_results['sampled_dishes_by_customer_idx'], axis=0)
    # plt.imshow(mc_avg)
    # plt.show()

    return sample_ibp_results


def sample_from_linear_gaussian(num_obs: int = 100,
                                num_true_features: int = 5,
                                indicator_sampling_str: str = 'categorical',
                                indicator_sampling_params: Dict[str, float] = None,
                                feature_prior_params: Dict[str, float] = None,
                                likelihood_params: Dict[str, float] = None) -> Dict[str, np.ndarray]:
    """
    Draw sample from Binary Linear-Gaussian model.

    :return:
        sampled_indicators: NumPy array with shape (seq_len,) of (integer) sampled classes
        linear_gaussian_samples_seq: NumPy array with shape (seq_len, obs_dim) of
                                binary linear-Gaussian samples
    """

    if feature_prior_params is None:
        feature_prior_params = {}

    if likelihood_params is None:
        likelihood_params = {'sigma_x': 1e-1}

    # Otherwise, use categorical or IBP to sample number of features
    if indicator_sampling_str not in {'categorical', 'IBP', 'GriffithsGhahramani'}:
        raise ValueError(f'Impermissible class sampling value: {indicator_sampling_str}')

    # # Unique case of generating Griffiths & Ghahramani data
    # if indicator_sampling_str == 'GriffithsGhahramani':
    #     sampled_data_result = sample_from_griffiths_ghahramani_2005(
    #         num_obs=num_obs,
    #         gaussian_likelihood_params=likelihood_params)
    #     sampled_data_result['indicator_sampling_str'] = indicator_sampling_str
    #
    #     indicator_sampling_descr_str = '{}_probs=[{}]'.format(
    #         indicator_sampling_str,
    #         ','.join([str(i) for i in sampled_data_result['indicator_sampling_params']['probs']]))
    #     sampled_data_result['indicator_sampling_descr_str'] = indicator_sampling_descr_str
    #     return sampled_data_result

    if indicator_sampling_str is None:
        indicator_sampling_params = dict()

    if indicator_sampling_str == 'categorical':

        # if probabilities per cluster aren't specified, go with uniform probabilities
        if 'probs' not in indicator_sampling_params:
            indicator_sampling_params['probs'] = np.ones(num_true_features) / num_true_features

        indicator_sampling_descr_str = '{}_probs={}'.format(
            indicator_sampling_str,
            list(indicator_sampling_params['probs']))
        indicator_sampling_descr_str = indicator_sampling_descr_str.replace(' ', '')

    elif indicator_sampling_str == 'IBP':
        if 'alpha' not in indicator_sampling_params:
            indicator_sampling_params['alpha'] = 3.98
        if 'beta' not in indicator_sampling_params:
            indicator_sampling_params['beta'] = 4.97
        indicator_sampling_descr_str = '{}_a={}_b={}'.format(
            indicator_sampling_str,
            indicator_sampling_params['alpha'],
            indicator_sampling_params['beta'])

    else:
        raise NotImplementedError

    if indicator_sampling_str == 'categorical':
        # num_gaussians = indicator_sampling_params['probs'].shape[0]
        sampled_indicators = np.random.binomial(
            n=1,
            p=indicator_sampling_params['probs'][np.newaxis, :],
            size=(num_obs, num_true_features))
    elif indicator_sampling_str == 'IBP':
        sampled_indicators = sample_ibp(
            num_mc_sample=1,
            num_customer=num_obs,
            num_true_features=num_true_features,
            alpha=indicator_sampling_params['alpha'],
            beta=indicator_sampling_params['beta'])['sampled_dishes_by_customer_idx'][0, :, :]
        # num_gaussians = np.argwhere(np.sum(sampled_indicators, axis=0) == 0.)[0, 0]
        sampled_indicators = sampled_indicators[:, :num_true_features]
    else:
        raise ValueError(f'Impermissible class sampling: {indicator_sampling_str}')

    gaussian_params = utils.data.synthetic.generate_gaussian_params_from_gaussian_prior(
        num_gaussians=num_true_features,
        **feature_prior_params)

    features = gaussian_params['means']
    obs_dim = features.shape[1]
    print("SHAPE CHECK:",sampled_indicators.shape, features.shape)
    obs_means = np.matmul(sampled_indicators, features)
    obs_cov = np.square(likelihood_params['sigma_x']) * np.eye(obs_dim)
    observations = np.array([
        np.random.multivariate_normal(
            mean=obs_means[obs_idx],
            cov=obs_cov)
        for obs_idx in range(num_obs)])

    sampled_data_result = dict(
        gaussian_params=gaussian_params,
        sampled_indicators=sampled_indicators,
        observations=observations,
        features=features,
        indicator_sampling_str=indicator_sampling_str,
        indicator_sampling_params=indicator_sampling_params,
        indicator_sampling_descr_str=indicator_sampling_descr_str,
        feature_prior_params=feature_prior_params,
        likelihood_params=likelihood_params,
    )

    return sampled_data_result


def analyze_asymptotics():
    # create directory
    exp_dir_path = '02_linear_gaussian'
    results_dir_path = os.path.join(exp_dir_path, 'analyze_asymptotics')
    os.makedirs(results_dir_path, exist_ok=True)

    feature_samplings = [
        # ('GriffithsGhahramani', dict()),
        # ('categorical', dict(probs=np.ones(5) / 5.)),
        # ('categorical', dict(probs=np.array([0.4, 0.25, 0.2, 0.1, 0.05]))),
        ('IBP', dict(alpha=1.17, beta=1.)),
        # ('IBP', dict(alpha=2.4, beta=1.)),
        # ('IBP', dict(alpha=5.98, beta=1.)),
    ]

    gaussian_cov_scaling: float = 0.3
    feature_prior_cov_scaling: float = 100.

    # num_customers_array = np.arange(10,10**4)
    # num_true_features_array = np.arange(10,11)
    num_customers_array = np.array([10, 50, 100, 200])
    num_true_features_array = np.unique(np.logspace(0,5,num=400).astype(int))
    # num_true_features_array = np.array([30,31,32,33])

    inference_alg_strs = [
        'R-IBP',
        # 'HMC-Gibbs',
        # 'Collapsed-Gibbs',
        # 'Doshi-Velez-Finite',
        # 'Doshi-Velez-Infinite',
        # 'Widjaja-Finite',
        # 'Widjaja-Infinite',
    ]
    hyperparams = [inference_alg_strs]

    num_inferred_features_per_num_obs_per_num_true_features_array = np.empty((num_customers_array.shape[0], num_true_features_array.shape[0]))

    # Launch inference for different numbers of observations and true features
    for (indicator_sampling, indicator_sampling_params), num_customers_idx, num_true_features_idx in \
            itertools.product(feature_samplings, range(num_customers_array.shape[0]), range(num_true_features_array.shape[0])):

        num_customers = num_customers_array[num_customers_idx]
        num_true_features = num_true_features_array[num_true_features_idx]

        print("NUM CUSTOMERS AND GT FEATURES:",num_customers, num_true_features)
        logging.info(f'Sampling: {indicator_sampling}, '
                     f'Number of Customers: {num_customers}, '
                     f'Number of True Features: {num_true_features}')

        # Sample the data
        sampled_linear_gaussian_data = sample_from_linear_gaussian(
            num_obs=num_customers,
            num_true_features=num_true_features,
            indicator_sampling_str=indicator_sampling,
            indicator_sampling_params=indicator_sampling_params,
            feature_prior_params=dict(gaussian_cov_scaling=gaussian_cov_scaling,
                                      feature_prior_cov_scaling=feature_prior_cov_scaling))

        # save dataset
        run_one_results_dir_path = os.path.join(
            results_dir_path,
            sampled_linear_gaussian_data['indicator_sampling_descr_str'],
            f'num_obs={num_customers}',
            f'num_true_features={num_true_features}')
        os.makedirs(run_one_results_dir_path, exist_ok=True)
        joblib.dump(sampled_linear_gaussian_data,
                    filename=os.path.join(run_one_results_dir_path, 'data.joblib'))

        # plot_linear_gaussian.plot_run_one_sample_from_linear_gaussian(
        #     features=sampled_linear_gaussian_data['features'],
        #     observations=sampled_linear_gaussian_data['observations'],
        #     plot_dir=run_one_results_dir_path)

        for inference_alg_str, in itertools.product(*hyperparams):
            alpha = indicator_sampling_params['alpha']
            beta = indicator_sampling_params['beta']

            # get # of inferred features for current # of observations & # of true features
            asymptotic_datapoint = eval(launch_run_one(
                exp_dir_path=exp_dir_path,
                run_one_results_dir_path=run_one_results_dir_path,
                inference_alg_str=inference_alg_str,
                alpha=alpha,
                beta=beta))
            # print("ASYMPTOTIC DATAPOINT:",(num_customers, num_true_features,asymptotic_datapoint))
            print("ASYMPTOTIC DATAPOINT:",asymptotic_datapoint,'\n')
            # save number of inferred features to list
            num_inferred_features_per_num_obs_per_num_true_features_array[num_customers_idx][num_true_features_idx] = asymptotic_datapoint[-1]
            np.save(results_dir_path+'/asymptotic_array_alpha_1.17_beta_1.npy', num_inferred_features_per_num_obs_per_num_true_features_array)
            # np.load('test3.npy')
            continue

    # generate plot of asymptotic # inferred features vs true features
    plot_asymp_inferred_feature_num_vs_true_feature_num(
        num_inferred_features_per_num_obs_per_num_true_features_array,
        plot_dir=results_dir_path)


def launch_run_one(exp_dir_path: str,
                   run_one_results_dir_path: str,
                   inference_alg_str: str,
                   alpha: float,
                   beta: float):

    # run_one_script_path = os.path.join(exp_dir_path, 'run_one_asymptotic.py')
    # run_one_script_path = './run_one_asymptotic.sh'
    run_one_script_path = os.path.join(exp_dir_path, 'run_one_asymptotic.sh')
    command_and_args = [
        # 'sbatch',
        run_one_script_path,
        run_one_results_dir_path,
        inference_alg_str,
        str(alpha),
        str(beta)]

    # TODO: Figure out where the logger is logging to
    logging.info(f'Launching ' + ' '.join(command_and_args))
    # subprocess.run(command_and_args)
    # asymptotic_datapoint = subprocess.run(command_and_args, capture_output=True).stdout #, encoding='UTF-8'
    asymptotic_datapoint = subprocess.check_output(command_and_args, encoding='UTF-8')
    logging.info(f'Launched ' + ' '.join(command_and_args))
    return asymptotic_datapoint

def plot_asymp_inferred_feature_num_vs_true_feature_num(num_inferred_features_per_num_obs_per_num_true_features: np.ndarray,
                                                        plot_dir: str):

    fig, axes = plt.subplots(nrows=1,
                             ncols=1,
                             figsize=(13, 4),
                             gridspec_kw={"width_ratios": [1, 0.1, 1, 0.1, 1]})

    ax = axes[0]
    sns.heatmap(
        data=num_inferred_features_per_num_obs_per_num_true_features,
        ax=ax,
        cbar_kws=dict(label='Number of Inferred Features'),
        cmap='jet',
        mask=np.isnan(num_inferred_features_per_num_obs_per_num_true_features),
        vmin=0,
        vmax=np.nanmax(num_inferred_features_per_num_obs_per_num_true_features),
        norm=LogNorm(),
    )
    plt.yscale('log')
    ax.set_xlabel('Number of True Features')
    ax.set_ylabel('Number of Observations')
    ax.set_title('Asymptotic Number of Inferred Features \nversus Number of True Features')

    plt.savefig(os.path.join(plot_dir, 'asymptotic_inferred_feature_num_vs_true_feature_num.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    analyze_asymptotics()
    logging.info('Finished.')
