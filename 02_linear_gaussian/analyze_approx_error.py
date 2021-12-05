"""
Evaluate the approximation error of the IBP against Infinite
Doshi-Velez variant.

Example usage:

02_prior/analyze_approx_error.py
"""
import argparse
import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


import utils.inference


def analyze_approx_error(args: argparse.Namespace):
    # create directory
    exp_dir_path = args.exp_dir_path
    results_dir_path = os.path.join(exp_dir_path, 'results')

    rows = load_all_datasets_all_alg_results(
        results_dir_path=results_dir_path)

    plt.rcParams.update({'font.size': 18})
    start_size = 5
    step_size = 5
    reconstruction_errors = []
    for row_idx, row in enumerate(rows):

        # if row_idx > 15:
        #     break

        total_num_obs = row['train_observations'].shape[0]
        for num_obs in range(start_size, total_num_obs + 1, step_size):
            print(f'Row idx: {row_idx}\tNum Obs: {num_obs}')
            first_train_obs = row['train_observations'][:num_obs]
            inference_alg_results = utils.inference.run_inference_alg(
                inference_alg_str='Doshi-Velez-Finite',
                observations=first_train_obs,
                model_str='linear_gaussian',
                gen_model_params=row['gen_model_params'])
            doshi_velez_dish_posteriors = inference_alg_results['dish_eating_posteriors']
            doshi_velez_features_after_last_obs = inference_alg_results['inference_alg'].features_after_last_obs()
            doshi_velex_similarity_matrix = doshi_velez_dish_posteriors @ doshi_velez_dish_posteriors.T

            ribp_dish_posteriors = row['dish_eating_posteriors'][:num_obs]
            ribp_similarity_matrix = ribp_dish_posteriors @ ribp_dish_posteriors.T

            # Subtract 1 because 0 based indexing; we want endpoint to be excluded
            # i.e. if we have 10 obs, we want features on 9th index
            ribp_features_after_last_obs = row['features_by_obs'][num_obs-1]

            doshi_velez_diff = first_train_obs - doshi_velez_dish_posteriors @ doshi_velez_features_after_last_obs
            doshi_velez_reconstruction_error = np.linalg.norm(doshi_velez_diff)
            ribp_diff = first_train_obs - ribp_dish_posteriors @ ribp_features_after_last_obs
            ribp_reconstruction_error = np.linalg.norm(ribp_diff)

            for inf_alg, sim_mat, recon_err in [('R-IBP', ribp_similarity_matrix, ribp_reconstruction_error),
                                                ('Doshi-Velez-Finite', doshi_velex_similarity_matrix, doshi_velez_reconstruction_error)]:
                reconstruction_errors.append({
                    'sampling_scheme': row['sampling_scheme'],
                    'sampling_dir_path': row['sampling_dir_path'],
                    'alpha': row['alpha'],
                    'beta': row['beta'],
                    'dataset': row['dataset'],
                    'num_obs': num_obs,
                    'inference_alg': inf_alg,
                    'similarity_matrix': sim_mat,
                    'reconstruction_error': recon_err,
                })

        reconstruction_errors_df = pd.DataFrame(reconstruction_errors)

        for sampling_scheme, recon_error_by_sampling_df in reconstruction_errors_df.groupby('sampling_scheme'):
            sns.lineplot(
                data=recon_error_by_sampling_df,
                x='num_obs',
                y='reconstruction_error',
                hue='inference_alg',)
            alpha = recon_error_by_sampling_df['alpha'].values[0]
            beta = recon_error_by_sampling_df['beta'].values[0]
            plt.title(rf'$\alpha={alpha}, \beta={beta}$')
            plt.xlabel('Number of Observations (n)')
            plt.ylabel(r'$||O_{\leq n} - Z_{\leq n} \, A_n ||_F$')
            plt.legend()
            plt.subplots_adjust(left=0.15)
            plt.subplots_adjust(bottom=0.15)
            sampling_results_dir_path = recon_error_by_sampling_df['sampling_dir_path'].values[0]
            plt.savefig(os.path.join(sampling_results_dir_path,
                                     f'reconstruction_error_vs_num_obs_{sampling_scheme}_cont_start={start_size}_step={step_size}.png'),
                        bbox_inches='tight',
                        dpi=300)
            # plt.show()
            plt.close()


def load_all_datasets_all_alg_results(results_dir_path) -> pd.DataFrame:
    rows = []
    sampling_dirs = [subdir for subdir in os.listdir(results_dir_path)]

    # Iterate through each sampling scheme directory
    for sampling_dir in sampling_dirs:
        sampling_dir_path = os.path.join(results_dir_path, sampling_dir)
        # Iterate through each sampled dataset
        dataset_dirs = [subdir for subdir in os.listdir(sampling_dir_path)
                        if os.path.isdir(os.path.join(sampling_dir_path, subdir))]
        for dataset_dir in dataset_dirs:
            dataset_dir_path = os.path.join(sampling_dir_path, dataset_dir)
            # Find all algorithms that were run
            inference_alg_dirs = [sub_dir for sub_dir in os.listdir(dataset_dir_path)
                                  if os.path.isdir(os.path.join(dataset_dir_path, sub_dir))]
            for inference_alg_dir in inference_alg_dirs:

                if not inference_alg_dir.startswith('R-IBP'):
                    continue

                inference_alg_dir_path = os.path.join(dataset_dir_path, inference_alg_dir)
                try:
                    stored_data = joblib.load(
                        os.path.join(inference_alg_dir_path, 'inference_alg_results.joblib'))
                except FileNotFoundError:
                    logging.info(f'Could not find results for {inference_alg_dir_path}.')
                    continue
                logging.info(f'Successfully loaded {inference_alg_dir_path} algorithm results.')

                new_row = dict(
                    sampling_scheme=sampling_dir,
                    sampling_dir_path=sampling_dir_path,
                    dataset=dataset_dir,
                    inference_alg=stored_data['inference_alg_str'],
                    alpha=stored_data['inference_alg_params']['IBP']['alpha'],
                    beta=stored_data['inference_alg_params']['IBP']['beta'],
                    dish_eating_posteriors=stored_data['inference_alg_results']['dish_eating_posteriors'],
                    features_by_obs=stored_data['inference_alg_results']['inference_alg'].features_by_obs(),
                    train_observations=stored_data['sampled_linear_gaussian_data']['train_observations'],
                    gen_model_params=stored_data['inference_alg_results']['gen_model_params'])

                del stored_data
                rows.append(new_row)

    # inf_algorithms_results_df = pd.DataFrame(
    #     rows,
    #     columns=['sampling', 'dataset', 'inference_alg', 'alpha',
    #              'beta', 'dish_eating_posteriors', 'features', 'train_observations'])

    # return inf_algorithms_results_df

    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir_path', type=str,
                        default='02_linear_gaussian',
                        help='Path to write plots and other results to.')
    args = parser.parse_args()
    analyze_approx_error(args=args)
    logging.info('Finished.')
