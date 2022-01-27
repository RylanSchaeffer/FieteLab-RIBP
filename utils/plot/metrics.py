from matplotlib.colors import LogNorm
import matplotlib as mpl
import matplotlib.cm
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


def compute_tidied_label(label: str) -> str:

    if label == 'runtime':
        tidied_ylabel = 'Runtime'
    elif label == 'negative_log_posterior_predictive':
        tidied_ylabel = 'Neg Log Posterior Predictive'
    elif label == 'reconstruction_error':
        tidied_ylabel = 'Reconstruction Error'
    else:
        tidied_ylabel = label


def plot_neg_log_posterior_predictive_vs_runtime_by_alg(inf_algs_results_df: pd.DataFrame,
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


def plot_recon_error_vs_runtime_by_alg(inf_algs_results_df: pd.DataFrame,
                                       plot_dir: str,
                                       title: str = None):
    for inf_alg_str, inf_alg_results_df in inf_algs_results_df.groupby(['inference_alg']):
        algs_runstimes_and_train_recon_error = inf_alg_results_df.agg({
            'runtime': ['mean', 'sem'],
            'reconstruction_error': ['mean', 'sem']
        })

        plt.errorbar(
            x=algs_runstimes_and_train_recon_error['runtime']['mean'],
            xerr=algs_runstimes_and_train_recon_error['runtime']['sem'],
            y=algs_runstimes_and_train_recon_error['reconstruction_error']['mean'],
            yerr=algs_runstimes_and_train_recon_error['reconstruction_error']['sem'],
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


def plot_score_all_params_violin_by_alg(inf_algs_results_df: pd.DataFrame,
                                        plot_dir: str,
                                        score: str = 'neg_log_posterior_predictive',
                                        title: str = None):
    assert score in inf_algs_results_df.columns.values

    sns.violinplot(x='inference_alg',
                   y=score,
                   data=inf_algs_results_df)

    tidied_ylabel = compute_tidied_label(label=score)
    plt.ylabel(tidied_ylabel)
    plt.yscale('log')
    plt.grid(visible=True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{score}_all_params_vs_inf_alg.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_score_best_by_alg(inf_algs_results_df: pd.DataFrame,
                           plot_dir: str,
                           score: str = 'neg_log_posterior_predictive'):

    assert score in inf_algs_results_df.columns.values

    best_score_inf_algs_results_df = inf_algs_results_df.groupby('inference_alg').agg({
        score: ['min']
    })[score]

    # Move inference_alg from index to a new column.
    best_score_inf_algs_results_df.reset_index(inplace=True)

    sns.catplot(x='inference_alg',
                y='min',
                data=best_score_inf_algs_results_df,
                jitter=False)

    tidied_ylabel = compute_tidied_label(label=score)
    plt.ylabel(tidied_ylabel)
    plt.yscale('log')
    # plt.legend()
    plt.grid(visible=True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{score}_best_vs_inf_alg.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_runtime_by_alpha_beta(inf_algs_results_df: pd.DataFrame,
                               plot_dir: str):

    inf_algs_results_df['alpha_rounded'] = np.round(inf_algs_results_df['alpha'],
                                                    decimals=3)

    inf_algs_results_df['beta_rounded'] = np.round(inf_algs_results_df['beta'],
                                                   decimals=3)

    g = sns.lineplot(
        data=inf_algs_results_df,
        x='alpha_rounded',
        y='runtime',
        hue='beta_rounded',
        legend='full',  # Ensures hue is treated as continuum & not binned.
        palette='rocket_r',
    )

    norm = plt.Normalize(0.,
                         inf_algs_results_df['beta_rounded'].max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap="rocket_r")
    sm.set_array([])
    g.get_legend().remove()
    g.figure.colorbar(sm, label=r'$\beta$')

    plt.xlabel(r'$\alpha$')
    plt.ylabel('Runtime')
    # plt.xscale('log')
    plt.yscale('log')
    plt.grid(visible=True, axis='both')

    plt.savefig(os.path.join(plot_dir,
                             f'runtime_by_alpha_beta.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
