import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import utils.plot


plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16  # was previously 22
sns.set_style("whitegrid")


def plot_analyze_all_algorithms_results(inf_algs_results_df: pd.DataFrame,
                                        plot_dir: str):

    title = '2016 Cancer Gene Expression'

    utils.plot.plot_neg_log_posterior_predictive_vs_runtime_by_alg(
        inf_algs_results_df=inf_algs_results_df,
        plot_dir=plot_dir,
        title=title)

    utils.plot.plot_recon_error_vs_runtime_by_alg(
        inf_algs_results_df=inf_algs_results_df,
        plot_dir=plot_dir,
        title=title)
