from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Union

from utils.numpy_helpers import compute_largest_dish_idx

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16  # was previously 22
sns.set_style("whitegrid")


def plot_neg_log_posterior_predictive_vs_runtime_by_alg(
        inf_algs_results_df: pd.DataFrame,
        plot_dir: str,
        title: str = None):

    for inf_alg_str, inf_alg_results_df in inf_algs_results_df.groupby(['inference_alg']):
        algs_runstimes_and_neg_log_pp = inf_alg_results_df.agg({
            'runtime': ['mean', 'sem'],
            'negative_log_posterior_predictive': ['mean', 'sem']
        })

        plt.errorbar(
            x=algs_runstimes_and_neg_log_pp['runtime']['mean'],
            xerr=algs_runstimes_and_neg_log_pp['runtime']['sem'],
            y=algs_runstimes_and_neg_log_pp['negative_log_posterior_predictive']['mean'],
            yerr=algs_runstimes_and_neg_log_pp['negative_log_posterior_predictive']['sem'],
            fmt='o',
            label=inf_alg_str)

    # plt.legend()
    lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    if title is not None:
        plt.title(title)
    plt.xlabel('Runtime (s)')
    plt.ylabel('Negative Log Posterior Predictive')
    plt.xscale('log')
    plt.yscale('log')
    # Make room at bottom
    plt.subplots_adjust(left=0.15)
    plt.subplots_adjust(bottom=0.15)
    # plt.tight_layout()
    plt.grid(visible=True, axis='both')
    plt.savefig(os.path.join(plot_dir,
                             f'negative_posterior_predictive_vs_runtime.png'),
                bbox_extra_artists=(lg,),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_recon_error_vs_runtime_by_alg(
        inf_algs_results_df: pd.DataFrame,
        plot_dir: str,
        title: str = None):

    for inf_alg_str, inf_alg_results_df in inf_algs_results_df.groupby(['inference_alg']):

        algs_runstimes_and_train_recon_error = inf_alg_results_df.agg({
            'runtime': ['mean', 'sem'],
            'training_reconstruction_error': ['mean', 'sem']
        })

        plt.errorbar(
            x=algs_runstimes_and_train_recon_error['runtime']['mean'],
            xerr=algs_runstimes_and_train_recon_error['runtime']['sem'],
            y=algs_runstimes_and_train_recon_error['training_reconstruction_error']['mean'],
            yerr=algs_runstimes_and_train_recon_error['training_reconstruction_error']['sem'],
            fmt='o',
            label=inf_alg_str)

    # plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    if title is not None:
        plt.title(title)
    plt.xlabel('Runtime (s)')
    plt.ylabel(r'$||X - Z A||_2^2$')
    plt.xscale('log')
    plt.grid(visible=True, axis='both')
    # plt.yscale('log')
    # Make room at bottom
    # plt.subplots_adjust(bottom=0.15)
    # Ensures things aren't cut off - maybe?
    # https://www.delftstack.com/howto/matplotlib/how-to-place-legend-outside-of-the-plot-in-matplotlib/
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'reconstruction_error_vs_runtime.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
