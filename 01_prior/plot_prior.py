import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import scipy.stats
import seaborn as sns

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16
sns.set_style("whitegrid")


alphas_color_map = {
    1.1: 'tab:blue',
    10.37: 'tab:orange',
    15.78: 'tab:purple',
    30.91: 'tab:green'
}

# Font setup
# sns.set(font_scale=1.2)
# plt.rcParams["font.family"] = "serif"
# plt.rcParams['font.serif'] = "Times New Roman"
# params = {'axes.labelsize': 'medium',
#           'axes.titlesize':'large',
#           'xtick.labelsize':'medium',
#           'ytick.labelsize':'medium'}
# pylab.rcParams.update(params)


def plot_analytical_vs_monte_carlo_mse(mse_df: pd.DataFrame,
                                       plot_dir: str):
    # combine alpha, beta into one column
    mse_df['alpha_beta'] = mse_df.agg(
        lambda row: rf"$\alpha$={row['alpha']}, $\beta$={row['beta']}",
        axis=1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    sns.lineplot(data=mse_df,
                 x='num_samples',
                 y='mse',
                 hue='alpha_beta',
                 ax=ax)

    # Remove legend title. Seaborn always uses "hue"
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'(Analytic - Monte Carlo$)^2$')
    ax.set_xlabel('Number of Monte Carlo Samples')
    # ax.set_ylabel(r'$\mathbb{E}_D[\sum_k (\mathbb{E}[N_{T, k}] - \frac{1}{S} \sum_{s=1}^S N_{T, k}^{(s)})^2]$')
    fig.savefig(os.path.join(plot_dir, f'mse_analytical_vs_monte_carlo.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_customer_dishes_analytical_vs_monte_carlo(sampled_dishes_by_customer_idx: np.ndarray,
                                                   analytical_dishes_by_customer_idx: np.ndarray,
                                                   alpha: float,
                                                   beta: float,
                                                   plot_dir: str):
    # plot customer assignments, comparing analytics versus monte carlo estimates
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    avg_sampled_dishes_by_customer = np.mean(
        sampled_dishes_by_customer_idx, axis=0)

    # identify highest non-zero dish index to truncate plot
    max_dish_idx = np.argwhere(np.sum(avg_sampled_dishes_by_customer, axis=0) > 0.)[-1, 0]
    avg_sampled_dishes_by_customer = avg_sampled_dishes_by_customer[:, :max_dish_idx]
    analytical_dishes_by_customer_idx = analytical_dishes_by_customer_idx[:, :max_dish_idx]

    # replace 0s with nans to allow for log scaling
    nan_idx = avg_sampled_dishes_by_customer == 0.
    avg_sampled_dishes_by_customer[nan_idx] = np.nan
    cutoff = np.nanmin(avg_sampled_dishes_by_customer)
    cutoff_idx = avg_sampled_dishes_by_customer < cutoff
    avg_sampled_dishes_by_customer[cutoff_idx] = np.nan

    ax = axes[0]
    sns.heatmap(avg_sampled_dishes_by_customer,
                ax=ax,
                mask=np.isnan(avg_sampled_dishes_by_customer),
                cmap='Spectral',
                vmin=cutoff,
                vmax=1.,
                norm=LogNorm())

    ax.set_title(rf'Monte Carlo ($\alpha=${alpha}, $\beta=${beta})')
    ax.set_ylabel(r'Customer Index')
    ax.set_xlabel(r'Dish Index')

    ax = axes[1]
    cutoff_idx = analytical_dishes_by_customer_idx < cutoff
    analytical_dishes_by_customer_idx[cutoff_idx] = np.nan
    sns.heatmap(analytical_dishes_by_customer_idx,
                ax=ax,
                mask=np.isnan(analytical_dishes_by_customer_idx),
                cmap='Spectral',
                norm=LogNorm(),
                vmin=cutoff,
                vmax=1.,
                )
    ax.set_title(rf'Analytical ($\alpha=${alpha}, $\beta$={beta})')
    ax.set_xlabel(r'Dish Index')

    # for some reason, on OpenMind, colorbar ticks disappear without calling plt.show() first
    plt.show()
    fig.savefig(os.path.join(plot_dir, f'customer_dishes_monte_carlo_vs_analytical_a={alpha}_b={beta}.png'),
                bbox_inches='tight',
                dpi=300)
    plt.close()


def plot_num_dishes_analytical_vs_monte_carlo(sampled_num_dishes_by_customer_idx: np.ndarray,
                                              analytical_num_dishes_by_customer: np.ndarray,
                                              alpha: float,
                                              beta: float,
                                              plot_dir: str):
    # plot customer assignments, comparing analytics versus monte carlo estimates
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    avg_sampled_num_tables_by_customer = np.mean(
        sampled_num_dishes_by_customer_idx, axis=0)

    # identify highest non-zero dish index to truncate plot
    max_dish_idx = np.argwhere(np.sum(avg_sampled_num_tables_by_customer, axis=0) > 0.)[-1, 0]
    avg_sampled_num_tables_by_customer = avg_sampled_num_tables_by_customer[:, :max_dish_idx]
    analytical_num_dishes_by_customer = analytical_num_dishes_by_customer[:, :max_dish_idx]

    # replace 0s with nans to allow for log scaling
    nan_idx = avg_sampled_num_tables_by_customer == 0.
    avg_sampled_num_tables_by_customer[nan_idx] = np.nan
    cutoff = np.nanmin(avg_sampled_num_tables_by_customer)
    cutoff_idx = avg_sampled_num_tables_by_customer < cutoff
    avg_sampled_num_tables_by_customer[cutoff_idx] = np.nan

    ax = axes[0]
    sns.heatmap(avg_sampled_num_tables_by_customer,
                ax=ax,
                mask=np.isnan(avg_sampled_num_tables_by_customer),
                cmap='Spectral',
                norm=LogNorm(),
                vmin=cutoff,
                vmax=1.,
                )

    ax.set_title(rf'Monte Carlo ($\alpha=${alpha}, $\beta=${beta})')
    ax.set_ylabel(r'Customer Index')
    ax.set_xlabel(r'Dish Index')

    ax = axes[1]
    cutoff_idx = analytical_num_dishes_by_customer < cutoff
    analytical_num_dishes_by_customer[cutoff_idx] = np.nan
    sns.heatmap(analytical_num_dishes_by_customer,
                ax=ax,
                mask=np.isnan(analytical_num_dishes_by_customer),
                cmap='Spectral',
                norm=LogNorm(),
                vmin=cutoff,
                vmax=1.,
                )
    ax.set_title(rf'Analytical ($\alpha=${alpha}, $\beta=${beta})')
    ax.set_xlabel(r'Dish Index')

    # for some reason, on OpenMind, colorbar ticks disappear without calling plt.show() first
    plt.show()
    fig.savefig(os.path.join(plot_dir, f'num_dishes_monte_carlo_vs_analytical_a={alpha}_b={beta}.png'),
                bbox_inches='tight',
                dpi=300)
    plt.close()


def plot_recursion_visualization(cum_analytical_dishes_by_customer_idx: np.ndarray,
                                 analytical_num_dishes_by_customer: np.ndarray,
                                 analytical_dishes_by_customer_idx: np.ndarray,
                                 alpha: float,
                                 beta: float,
                                 plot_dir: str,
                                 cutoff: float = 1e-4):
    fig, axes = plt.subplots(nrows=1,
                             ncols=5,
                             figsize=(15, 4),
                             gridspec_kw={"width_ratios": [1.5, 0.3, 1.5, 0.3, 1.5]},
                             sharex=True)

    # Font setup
    # sns.set(font_scale=2)
    # sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams['font.serif'] = "Times New Roman"
    # params = {'xtick.labelsize':20,
    #           'ytick.labelsize':20}
    # pylab.rcParams.update(params)
    # plt.rc('font',family = 'sans-serif'

    ax = axes[0]
    cum_analytical_dishes_by_customer_idx[cum_analytical_dishes_by_customer_idx < cutoff] = np.nan

    # figure out which dish falls below cutoff. if no dishes are below threshold, need to set
    max_dish_idx = np.argmax(np.nansum(cum_analytical_dishes_by_customer_idx, axis=0) < cutoff)
    if max_dish_idx == 0 and np.nansum(cum_analytical_dishes_by_customer_idx, axis=0)[max_dish_idx] >= cutoff:
        max_dish_idx = cum_analytical_dishes_by_customer_idx.shape[1]
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':'Times New Roman'})
    sns.heatmap(
        data=cum_analytical_dishes_by_customer_idx[:, :max_dish_idx],
        ax=ax,
        cbar_kws=dict(label=r'$\sum_{n^{\prime} = 1}^{n-1} p(z_{n^{\prime} k} = 1)$'),
        cmap='Spectral',
        mask=np.isnan(cum_analytical_dishes_by_customer_idx[:, :max_dish_idx]),
        vmin=cutoff,
        vmax=np.nanmax(cum_analytical_dishes_by_customer_idx),
        norm=LogNorm()
    )
    ax.set_xlabel('Dish Index', fontname='Times New Roman',fontsize=15)
    ax.set_ylabel('Customer Index', fontname='Times New Roman', fontsize=15)
    ax.set_title('Running Sum of Prev.\n Customers\' Dish Distributions', fontname='Times New Roman', fontsize=15)

    # necessary to make space for colorbar text
    axes[1].axis('off')

    ax = axes[2]
    analytical_num_dishes_by_customer[analytical_num_dishes_by_customer < cutoff] = np.nan
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':'Times New Roman'})
    sns.heatmap(
        data=analytical_num_dishes_by_customer[:, :max_dish_idx],
        ax=ax,
        cbar_kws=dict(label=r'$p(\Lambda_n = \ell)$'),
        cmap='Spectral',
        mask=np.isnan(analytical_num_dishes_by_customer[:, :max_dish_idx]),
        vmin=cutoff,
        vmax=np.nanmax(cum_analytical_dishes_by_customer_idx),
        # vmax=1.,
        norm=LogNorm()
    )
    ax.set_title('Distribution over\nTotal Number of Dishes', fontname='Times New Roman',fontsize=15)
    ax.set_xlabel('Dish Index', fontname='Times New Roman',fontsize=15)

    # necessary to make space for colorbar text
    axes[3].axis('off')

    ax = axes[4]
    analytical_dishes_by_customer_idx[analytical_dishes_by_customer_idx < cutoff] = np.nan
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':'Times New Roman'})
    sns.heatmap(
        data=analytical_dishes_by_customer_idx[:, :max_dish_idx],
        ax=ax,
        cbar_kws=dict(label='$p(z_{nk}=1)$'),
        cmap='Spectral',
        mask=np.isnan(analytical_dishes_by_customer_idx[:, :max_dish_idx]),
        vmin=cutoff,
        vmax=np.nanmax(cum_analytical_dishes_by_customer_idx),
        # vmax=1.,
        norm=LogNorm(),
    )
    ax.set_title('New Customer\'s\n Dish Distribution', fontname='Times New Roman',fontsize=15)
    ax.set_xlabel('Dish Index', fontname='Times New Roman',fontsize=15)

    # for some reason, on OpenMind, colorbar ticks disappear without calling plt.show() first
    plt.show()
    fig.savefig(os.path.join(plot_dir, f'ibp_recursion_a={alpha}_b={beta}.png'),
                bbox_inches='tight',
                dpi=300)
    plt.close()
