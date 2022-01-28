import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

import utils.plot.metrics
import utils.plot.linear_gaussian

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16  # was previously 22
sns.set_style("whitegrid")


def plot_analyze_all_algorithms_results(inf_algs_results_df: pd.DataFrame,
                                        plot_dir: str):

    title = '2016 Cancer Gene Expression'

    # Plot only R-IBP alpha/beta runtime tradeoff
    ribp_algs_results_df = inf_algs_results_df[inf_algs_results_df['inference_alg'] == 'R-IBP']
    utils.plot.metrics.plot_runtime_by_alpha_beta(
        inf_algs_results_df=ribp_algs_results_df,
        plot_dir=plot_dir)

    scores = ['runtime', 'negative_log_posterior_predictive', 'reconstruction_error']
    for score in scores:
        best_beta_all_dir = os.path.join(plot_dir, f'{score}_best_beta=all')
        os.makedirs(best_beta_all_dir, exist_ok=True)
        utils.plot.metrics.plot_score_best_by_alg(
            inf_algs_results_df,
            score=score,
            plot_dir=best_beta_all_dir,
            title=title,
        )

        best_beta_one_dir = os.path.join(plot_dir, f'{score}_best_beta=1')
        os.makedirs(best_beta_one_dir, exist_ok=True)
        utils.plot.metrics.plot_score_best_by_alg(
            inf_algs_results_df[inf_algs_results_df['beta'] == 1.],
            score=score,
            plot_dir=best_beta_one_dir,
            title=title,
        )

        all_dir = os.path.join(plot_dir, f'{score}_all')
        os.makedirs(all_dir, exist_ok=True)
        utils.plot.metrics.plot_score_all_params_violin_by_alg(
            inf_algs_results_df=inf_algs_results_df,
            score=score,
            plot_dir=all_dir,
            title=title,
        )

    # utils.plot.linear_gaussian.plot_neg_log_posterior_predictive_by_linear_gaussian_parameters(
    #     inf_algs_results_df=ribp_algs_results_df,
    #     plot_dir=plot_dir,
    #     title=title)
    #
    # utils.plot.linear_gaussian.plot_reconst_error_by_linear_gaussian_parameters(
    #     inf_algs_results_df=ribp_algs_results_df,
    #     plot_dir=plot_dir,
    #     title=title)

    utils.plot.metrics.plot_neg_log_posterior_predictive_vs_runtime_by_alg(
        inf_algs_results_df=inf_algs_results_df,
        plot_dir=plot_dir,
        title=title)

    utils.plot.metrics.plot_recon_error_vs_runtime_by_alg(
        inf_algs_results_df=inf_algs_results_df,
        plot_dir=plot_dir,
        title=title)
