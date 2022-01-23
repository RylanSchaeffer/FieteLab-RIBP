import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16  # was previously 22
sns.set_style("whitegrid")


def plot_neg_log_posterior_predictive_by_linear_gaussian_parameters(inf_algs_results_df: pd.DataFrame,
                                                                    plot_dir: str,
                                                                    title: str = None):
    sns.scatterplot(
        data=inf_algs_results_df,
        x='likelihood_cov_scaling',
        y='feature_cov_scaling',
        hue='negative_log_posterior_predictive',
        # legend='full',  # necessary to force seaborn to not try binning based on hue
        palette='RdBu',
    )

    norm = plt.Normalize(inf_algs_results_df['negative_log_posterior_predictive'].min(),
                         inf_algs_results_df['negative_log_posterior_predictive'].max())
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax = plt.gca()
    ax.get_legend().remove()
    ax.figure.colorbar(sm)

    if title is not None:
        plt.title(title)
    plt.ylabel(r'$\sigma_A^2$')
    plt.xlabel(r'$\sigma_o^2$')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(visible=True, axis='both')
    plt.savefig(os.path.join(plot_dir,
                             f'negative_posterior_predictive_by_cov_scalings.png'),
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()


def plot_reconst_error_by_linear_gaussian_parameters(inf_algs_results_df: pd.DataFrame,
                                                     plot_dir: str,
                                                     title: str = None):

    sns.scatterplot(
        data=inf_algs_results_df,
        x='likelihood_cov_scaling',
        y='feature_cov_scaling',
        hue='reconstruction_error')

    if title is not None:
        plt.title(title)
    plt.ylabel(r'$\sigma_A^2$')
    plt.xlabel(r'$\sigma_o^2$')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(visible=True, axis='both')
    plt.savefig(os.path.join(plot_dir,
                             f'negative_posterior_predictive_by_cov_scalings.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
