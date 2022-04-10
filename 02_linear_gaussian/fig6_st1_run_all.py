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

# import plot_linear_gaussian
import utils.data.synthetic


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
    data_dimensions_array = np.array([2, 10, 20])

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
        alpha_beta_datadim_string = 'alpha_' + str(alpha) + '_beta_' + str(beta) + '_datadim_' + str(data_dim)

        print("ON DATASET", dataset_idx, "WITH ALPHA, BETA, DATA DIM", alpha, beta, data_dim)
        logging.info(f'Sampling: {indicator_sampling}, Dataset Index: {dataset_idx}')
        sampled_linear_gaussian_data = utils.data.synthetic.sample_from_linear_gaussian(
            num_obs=num_customers,
            indicator_sampling_str=indicator_sampling,
            indicator_sampling_params=indicator_sampling_params,
            feature_prior_params=dict(gaussian_dim=data_dim,
                                      gaussian_cov_scaling=gaussian_cov_scaling,
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

        for inference_alg_str, in itertools.product(*hyperparams):  # only use RIBP
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

        if dataset_idx == num_datasets - 1:
            print("NUMBER OF DATASETS TO AVERAGE:", len(melted_data_to_plot_for_all_datasets))
            if need_initialize_dataframe:
                data_to_plot = pd.concat(melted_data_to_plot_for_all_datasets)
                need_initialize_dataframe = False
            else:
                results_for_all_datasets = pd.concat(melted_data_to_plot_for_all_datasets)
                data_to_plot = pd.concat([data_to_plot, results_for_all_datasets])

            # Save results
            data_to_plot.to_pickle(results_dir_path + '/data_to_plot_' + alpha_beta_datadim_string + '.pkl')
            print("DATA SAVED TO:", results_dir_path + '/data_to_plot_' + alpha_beta_datadim_string + '.pkl')

        # Plot result
        if data_dimension_idx == data_dimensions_array.shape[0] - 1 and dataset_idx == num_datasets - 1:
            plot_dir_path = os.path.join(results_dir_path,
                                         alpha_beta_datadim_string)
            os.makedirs(plot_dir_path, exist_ok=True)
            plot_avg_feature_ratio_by_num_obs_using_poisson_rates(data_to_plot=data_to_plot,
                                                                  plot_dir=plot_dir_path)
            # alpha_beta_datadim_string=alpha_beta_datadim_string)
            print("FIGURE SAVED TO:", plot_dir_path)


def plot_avg_feature_ratio_by_num_obs_using_poisson_rates(data_to_plot: pd.DataFrame,
                                                          # std_data_to_plot: pd.DataFrame,
                                                          plot_dir: str):
    # alpha_beta_datadim_string: str):
    fig, ax = plt.subplots(figsize=(10, 8))
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
    print("FIGURE SAVED TO:", plot_dir + '/fig6_st1_plot.png')


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
    logging.info(f'Launching ' + ' '.join(command_and_args))
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
