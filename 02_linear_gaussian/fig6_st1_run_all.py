"""
Launch run_one.py with each configuration of the hyperparameters.

Example usage:

02_linear_gaussian/run_all.py
"""

import itertools
import joblib
import logging
import numpy as np
import pandas as pd
import os
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt

import plot_linear_gaussian
from typing import Dict, Union
import utils.data.synthetic

def generate_gaussian_params_from_gaussian_prior(num_gaussians: int = 3,
                                                 gaussian_dim: int = 2,
                                                 feature_prior_cov_scaling: float = 3.,
                                                 gaussian_cov_scaling: float = 0.3):
    # sample Gaussians' means from prior = N(0, rho * I)
    means = np.random.multivariate_normal(
        mean=np.zeros(gaussian_dim),
        cov=feature_prior_cov_scaling * np.eye(gaussian_dim),
        size=num_gaussians)

    # all Gaussians have same covariance
    cov = gaussian_cov_scaling * np.eye(gaussian_dim)
    covs = np.repeat(cov[np.newaxis, :, :],
                     repeats=num_gaussians,
                     axis=0)

    mixture_of_gaussians = dict(means=means, covs=covs)

    return mixture_of_gaussians


def sample_from_linear_gaussian(num_obs: int = 100,
                                data_dim: int=2,
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

    if indicator_sampling_str is None:
        indicator_sampling_params = dict()

    if indicator_sampling_str == 'categorical':

        # if probabilities per cluster aren't specified, go with uniform probabilities
        if 'probs' not in indicator_sampling_params:
            indicator_sampling_params['probs'] = np.ones(5) / 5

        indicator_sampling_descr_str = '{}_probs={}'.format(
            indicator_sampling_str,
            list(indicator_sampling_params['probs']))
        indicator_sampling_descr_str = indicator_sampling_descr_str.replace(' ', '')

    elif indicator_sampling_str == 'IBP':
        if 'alpha' not in indicator_sampling_params:
            indicator_sampling_params['alpha'] = 3.98
        if 'beta' not in indicator_sampling_params:
            indicator_sampling_params['beta'] = 4.97
        indicator_sampling_params['datadim'] = data_dim
        indicator_sampling_descr_str = '{}_a={}_b={}_datadim={}'.format(
            indicator_sampling_str,
            indicator_sampling_params['alpha'],
            indicator_sampling_params['beta'],
            indicator_sampling_params['datadim'],)

    else:
        raise NotImplementedError

    if indicator_sampling_str == 'categorical':
        num_gaussians = indicator_sampling_params['probs'].shape[0]
        sampled_indicators = np.random.binomial(
            n=1,
            p=indicator_sampling_params['probs'][np.newaxis, :],
            size=(num_obs, num_gaussians))
    elif indicator_sampling_str == 'IBP':
        sampled_indicators = utils.data.synthetic.sample_ibp(
            num_mc_sample=1,
            num_customer=num_obs,
            alpha=indicator_sampling_params['alpha'],
            beta=indicator_sampling_params['beta'])['sampled_dishes_by_customer_idx'][0, :, :]
        num_gaussians = np.argwhere(np.sum(sampled_indicators, axis=0) == 0.)[0, 0]
        sampled_indicators = sampled_indicators[:, :num_gaussians]
    else:
        raise ValueError(f'Impermissible class sampling: {indicator_sampling_str}')

    gaussian_params = generate_gaussian_params_from_gaussian_prior(
        num_gaussians=num_gaussians,
        gaussian_dim=data_dim,
        **feature_prior_params)

    features = gaussian_params['means']
    obs_dim = features.shape[1]
    print("DATA OBS DIM:",obs_dim)
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


def run_all():
    # create directory
    exp_dir_path = '02_linear_gaussian'
    results_dir_path = os.path.join(exp_dir_path, 'results')
    os.makedirs(results_dir_path, exist_ok=True)

    feature_samplings = [
        # ('GriffithsGhahramani', dict()),
        # ('categorical', dict(probs=np.ones(5) / 5.)),
        # ('categorical', dict(probs=np.array([0.4, 0.25, 0.2, 0.1, 0.05]))),
        ('IBP', dict(alpha=1.17, beta=1.)),
        ('IBP', dict(alpha=2.4, beta=1.)),
        ('IBP', dict(alpha=5.98, beta=1.)),
    ]

    num_datasets = 10
    gaussian_cov_scaling: float = 0.3
    feature_prior_cov_scaling: float = 100.
    num_customers = 100
    # data_dimensions_array = np.array([2,3])
    data_dimensions_array = np.array([2,10,20])

    melted_data_to_plot_for_all_datasets = []

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

    need_initialize_dataframe = True

    # generate several datasets and independently launch inference
    for (indicator_sampling, indicator_sampling_params), data_dimension_idx, dataset_idx in \
            itertools.product(feature_samplings, range(data_dimensions_array.shape[0]), range(num_datasets)):

        alpha = indicator_sampling_params['alpha']
        beta = indicator_sampling_params['beta']
        data_dim = data_dimensions_array[data_dimension_idx]
        alpha_beta_datadim_string = 'alpha_'+str(alpha)+'_beta_'+str(beta)+'_datadim_'+str(data_dim)

        print("ON DATASET",dataset_idx,"WITH ALPHA, BETA, DATA DIM",alpha, beta, data_dim)
        logging.info(f'Sampling: {indicator_sampling}, Dataset Index: {dataset_idx}')
        sampled_linear_gaussian_data = sample_from_linear_gaussian(
            num_obs=num_customers,
            data_dim=data_dim,
            indicator_sampling_str=indicator_sampling,
            indicator_sampling_params=indicator_sampling_params,
            feature_prior_params=dict(gaussian_cov_scaling=gaussian_cov_scaling,
                                      feature_prior_cov_scaling=feature_prior_cov_scaling))

        # save dataset
        run_one_results_dir_path = os.path.join(
            results_dir_path,
            sampled_linear_gaussian_data['indicator_sampling_descr_str'],
            f'dataset={dataset_idx}')
        os.makedirs(run_one_results_dir_path, exist_ok=True)
        joblib.dump(sampled_linear_gaussian_data,
                    filename=os.path.join(run_one_results_dir_path,
                                          f'data.joblib'))

        # plot_linear_gaussian.plot_run_one_sample_from_linear_gaussian(
        #     features=sampled_linear_gaussian_data['features'],
        #     observations=sampled_linear_gaussian_data['observations'],
        #     plot_dir=run_one_results_dir_path)

        for inference_alg_str, in itertools.product(*hyperparams): # only use RIBP
            run_one_melted_data_path = str(launch_run_one(
                exp_dir_path=exp_dir_path,
                run_one_results_dir_path=run_one_results_dir_path,
                inference_alg_str=inference_alg_str,
                alpha=alpha,
                beta=beta,
                data_dim=data_dim)).strip('\n')
            # print("MELTED DATA TO PLOT FOR RUN ONE SAVED AT",run_one_melted_data_path)
            run_one_melted_data_to_plot = pd.read_pickle(run_one_melted_data_path)
            melted_data_to_plot_for_all_datasets.append(run_one_melted_data_to_plot)
            continue

        if dataset_idx == num_datasets-1:
            print("NUMBER OF DATASETS TO AVERAGE:",len(melted_data_to_plot_for_all_datasets))
            if need_initialize_dataframe:
                data_to_plot = pd.concat(melted_data_to_plot_for_all_datasets)
                need_initialize_dataframe = False
            else:
                results_for_all_datasets = pd.concat(melted_data_to_plot_for_all_datasets)
                data_to_plot = pd.concat([data_to_plot, results_for_all_datasets])

            # Save results
            data_to_plot.to_pickle(results_dir_path+'/data_to_plot_'+alpha_beta_datadim_string+'.pkl')
            print("DATA SAVED TO:",results_dir_path+'/data_to_plot_'+alpha_beta_datadim_string+'.pkl')

        # Plot result
        if data_dimension_idx==data_dimensions_array.shape[0]-1 and dataset_idx == num_datasets-1:
            plot_dir_path = os.path.join(results_dir_path,
                                         alpha_beta_datadim_string)
            os.makedirs(plot_dir_path, exist_ok=True)
            plot_avg_feature_ratio_by_num_obs_using_poisson_rates(data_to_plot=data_to_plot,
                                                                 plot_dir=plot_dir_path)
                                                                 # alpha_beta_datadim_string=alpha_beta_datadim_string)
            print("FIGURE SAVED TO:",plot_dir_path)


def plot_avg_feature_ratio_by_num_obs_using_poisson_rates(data_to_plot: pd.DataFrame,
                                                         # std_data_to_plot: pd.DataFrame,
                                                         plot_dir: str):
                                                         # alpha_beta_datadim_string: str):
    fig, ax = plt.subplots(figsize=(10,8))
    sns.set_style("darkgrid")
    g = sns.lineplot(x='obs_idx', y='dish_ratio', data=data_to_plot,
                     hue='data_dim', ci='sd', ax=ax, legend='full')

    # Remove "quantity" from legend title
    # see https://stackoverflow.com/questions/51579215/remove-seaborn-lineplot-legend-title
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles[1:], labels=labels[1:])

    plt.grid()
    plt.xlabel('Number of Observations')
    plt.ylabel('Num Inferred Features / Num True Features')
    plt.ylim(bottom=0.)
    g.get_legend().set_title('Data Dimension')
    plt.savefig(os.path.join(plot_dir, 'fig6_st1_plot.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print("FIGURE SAVED TO:",plot_dir+'/fig6_st1_plot.png')


def launch_run_one(exp_dir_path: str,
                   run_one_results_dir_path: str,
                   inference_alg_str: str,
                   alpha: float,
                   beta: float,
                   data_dim: int):

    run_one_script_path = os.path.join(exp_dir_path, f'fig6_st1_run_one.sh')
    command_and_args = [
        # 'sbatch',
        run_one_script_path,
        run_one_results_dir_path,
        inference_alg_str,
        str(alpha),
        str(beta),
        str(data_dim)]

    # TODO: Figure out where the logger is logging to
    # logging.info(f'Launching ' + ' '.join(command_and_args))
    return subprocess.check_output(command_and_args, encoding='UTF-8')
    # subprocess.run(command_and_args)
    # logging.info(f'Launched ' + ' '.join(command_and_args))


if __name__ == '__main__':
    run_all()

    # If only generating the figure:
    # data_to_plot = pd.read_pickle('/om2/user/gkml/FieteLab-RIBP/02_linear_gaussian/results/data_to_plot_alpha_5.98_beta_1.0_datadim_20.pkl')
    # plot_dir = '/om2/user/gkml/FieteLab-RIBP/02_linear_gaussian/results'
    # plot_avg_feature_ratio_by_num_obs_using_poisson_rates(data_to_plot=data_to_plot,
    #                                                       plot_dir=plot_dir)
    logging.info('Finished.')
