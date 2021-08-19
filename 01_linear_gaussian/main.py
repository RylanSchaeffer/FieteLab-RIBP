from itertools import product
import joblib
import os
from timeit import default_timer as timer

from exp_01_ibp_known_likelihood.plot import *
import utils.data
import utils.inference
import utils.metrics
import utils.plot


def main():
    plot_dir = 'exp_01_ibp_known_likelihood/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)

    num_datasets = 10
    inference_algs_results_by_dataset_idx = {}
    sampled_factor_analysis_results_by_dataset_idx = {}

    # generate lots of datasets and record performance for each
    for dataset_idx in range(num_datasets):
        print(f'Dataset Index: {dataset_idx}')
        dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_inference_algs_results, sampled_dataset_factor_analysis_results = run_one_dataset(
            dataset_dir=dataset_dir)
        inference_algs_results_by_dataset_idx[dataset_idx] = dataset_inference_algs_results
        sampled_factor_analysis_results_by_dataset_idx[dataset_idx] = sampled_dataset_factor_analysis_results

    utils.plot.plot_inference_algs_comparison(
        plot_dir=plot_dir,
        inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx,
        dataset_by_dataset_idx=sampled_factor_analysis_results_by_dataset_idx)

    print('Successfully completed Exp 01 Known Likelihoods')


def run_one_dataset(dataset_dir,
                    num_gaussians: int = 3,
                    gaussian_cov_scaling: float = 0.3,
                    gaussian_mean_prior_cov_scaling: float = 6.):

    # sample data
    beta_a, beta_b = 5, 6
    sampled_dataset_factor_analysis_results = utils.data.sample_sequence_from_factor_analysis(
        seq_len=250,
        obs_dim=2,  # TODO: restore to 25
        max_num_features=10000,
        weight_mean=1.,
        weight_variance=1e-20,  # effectively set all weights to 1 i.e. no noise
        obs_variance=0.0675,
        beta_a=beta_a,
        beta_b=beta_b)

    possible_inference_params = dict(
        alpha=[beta_a],
        beta=[beta_b])

    inference_alg_strs = [
        'R-IBP',
    ]

    inference_algs_results = {}
    for inference_alg_str in inference_alg_strs:
        inference_alg_results = run_and_plot_inference_alg(
            sampled_factor_analysis_results=sampled_dataset_factor_analysis_results,
            inference_alg_str=inference_alg_str,
            possible_inference_params=possible_inference_params,
            plot_dir=dataset_dir)
        inference_algs_results[inference_alg_str] = inference_alg_results
    return inference_algs_results, sampled_dataset_factor_analysis_results


def run_and_plot_inference_alg(sampled_factor_analysis_results,
                               inference_alg_str,
                               possible_inference_params,
                               plot_dir):

    inference_alg_plot_dir = os.path.join(plot_dir, inference_alg_str)
    os.makedirs(inference_alg_plot_dir, exist_ok=True)
    num_clusters_by_concentration_param = {}
    scores_by_concentration_param = {}
    runtimes_by_concentration_param = {}

    for alpha, beta in product(possible_inference_params['alpha'],
                               possible_inference_params['beta']):

        inference_params_str = f'a={np.round(alpha, 2)}_b={np.round(alpha, 2)}'

        inference_params = dict(
            alpha=alpha,
            beta=beta)

        inference_alg_results_params_path = os.path.join(
            inference_alg_plot_dir,
            f'results_{inference_params_str}.joblib')

        # if results do not exist, generate
        if not os.path.isfile(inference_alg_results_params_path):

            # run inference algorithm
            # time using timer because https://stackoverflow.com/a/25823885/4570472
            start_time = timer()
            inference_alg_particular_params_results = utils.inference.run_inference_alg(
                inference_alg_str=inference_alg_str,
                observations=sampled_factor_analysis_results['observations_seq'],
                inference_alg_params=inference_params,
                likelihood_model='multivariate_normal',
                learning_rate=1e0,
                likelihood_known=True,
                likelihood_params=dict(mean=sampled_factor_analysis_results['features'],
                                       cov=sampled_factor_analysis_results['obs_covariance']))

            # record elapsed time
            stop_time = timer()
            runtime = stop_time - start_time

            # # record scores
            # scores, pred_cluster_labels = utils.metrics.score_predicted_clusters(
            #     true_cluster_labels=sampled_factor_analysis_results['assigned_table_seq'],
            #     table_assignment_posteriors=inference_alg_particular_params_results['table_assignment_posteriors'])

            # # count number of clusters
            # num_clusters = len(np.unique(pred_cluster_labels))

            # write to disk and delete
            data_to_store = dict(
                # num_clusters=num_clusters,
                # scores=scores,
                sampled_factor_analysis_results=sampled_factor_analysis_results,
                inference_alg_particular_params_results=inference_alg_particular_params_results,
                runtime=runtime)

            joblib.dump(data_to_store,
                        filename=inference_alg_results_params_path)
            del inference_alg_particular_params_results
            del data_to_store

        # read results from disk
        stored_data = joblib.load(
            inference_alg_results_params_path)

        plot_inference_results(
            sampled_factor_analysis_results=stored_data['sampled_factor_analysis_results'],
            inference_results=stored_data['inference_alg_particular_params_results'],
            inference_alg_str=inference_alg_str,
            inference_params=inference_params,
            plot_dir=inference_alg_plot_dir)

        # num_clusters_by_concentration_param[inference_params_str] = stored_data[
        #     'num_clusters']
        # scores_by_concentration_param[inference_params_str] = stored_data[
        #     'scores']
        runtimes_by_concentration_param[inference_params_str] = stored_data[
            'runtime']

        print('Finished {} inference params: {}'.format(inference_alg_str, inference_params))

    inference_alg_particular_params_results = dict(
        num_clusters_by_param=num_clusters_by_concentration_param,
        scores_by_param=pd.DataFrame(scores_by_concentration_param).T,
        runtimes_by_param=runtimes_by_concentration_param,
    )

    return inference_alg_particular_params_results


if __name__ == '__main__':
    main()
