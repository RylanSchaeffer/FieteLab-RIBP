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

    utils.plot.metrics.plot_score_best_by_alg(
        inf_algs_results_df,
        score='runtime',
        plot_dir=plot_dir,
    )

    utils.plot.metrics.plot_score_all_params_violin_by_alg(
        inf_algs_results_df=inf_algs_results_df,
        score='runtime',
        plot_dir=plot_dir,
    )

    utils.plot.linear_gaussian.plot_neg_log_posterior_predictive_by_linear_gaussian_parameters(
        inf_algs_results_df=inf_algs_results_df[inf_algs_results_df['inference_alg'] == 'R-IBP'],
        plot_dir=plot_dir,
        title=title)

    utils.plot.linear_gaussian.plot_reconst_error_by_linear_gaussian_parameters(
        inf_algs_results_df=inf_algs_results_df[inf_algs_results_df['inference_alg'] == 'R-IBP'],
        plot_dir=plot_dir,
        title=title)

    utils.plot.metrics.plot_neg_log_posterior_predictive_vs_runtime_by_alg(
        inf_algs_results_df=inf_algs_results_df,
        plot_dir=plot_dir,
        title=title)

    utils.plot.metrics.plot_recon_error_vs_runtime_by_alg(
        inf_algs_results_df=inf_algs_results_df,
        plot_dir=plot_dir,
        title=title)
