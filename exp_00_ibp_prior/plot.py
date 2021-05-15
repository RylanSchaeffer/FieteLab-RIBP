import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats
import seaborn as sns


alphas_color_map = {
    1.1: 'tab:blue',
    10.78: 'tab:orange',
    15.37: 'tab:purple',
    30.91: 'tab:green'
}


def plot_analytical_vs_monte_carlo_mse(means_per_num_samples_per_alpha,
                                       sems_per_num_samples_per_alpha,
                                       num_reps,
                                       plot_dir):

    alphas = list(sems_per_num_samples_per_alpha.keys())

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    for alpha in alphas:
        ax.errorbar(x=list(means_per_num_samples_per_alpha[alpha].keys()),
                    y=list(means_per_num_samples_per_alpha[alpha].values()),
                    yerr=list(sems_per_num_samples_per_alpha[alpha].values()),
                    label=rf'$\alpha$={alpha}',
                    c=alphas_color_map[alpha])
    ax.legend(title=f'Num Repeats: {num_reps}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$(Analytic - Monte Carlo Estimate)^2$')
    # ax.set_ylabel(r'$\mathbb{E}_D[\sum_k (\mathbb{E}[N_{T, k}] - \frac{1}{S} \sum_{s=1}^S N_{T, k}^{(s)})^2]$')
    ax.set_xlabel('Number of Monte Carlo Samples')
    fig.savefig(os.path.join(plot_dir, f'crp_expected_mse.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_indian_buffet_dish_dist_by_customer_num(analytical_dish_distribution_poisson_rate_by_alpha_by_T,
                                                 plot_dir):
    # plot how the CRT table distribution changes for T customers
    alphas = list(analytical_dish_distribution_poisson_rate_by_alpha_by_T.keys())
    T = len(analytical_dish_distribution_poisson_rate_by_alpha_by_T[alphas[0]])
    customer_idx = 1 + np.arange(T)
    cmap = plt.get_cmap('jet_r')

    for alpha in alphas:
        for t in customer_idx:
            max_val = 2 * analytical_dish_distribution_poisson_rate_by_alpha_by_T[alpha][T]
            x = np.arange(0, max_val)
            y = scipy.stats.poisson.pmf(x, mu=analytical_dish_distribution_poisson_rate_by_alpha_by_T[alpha][t])
            plt.plot(x,
                     y,
                     # label=f'T={t}',
                     color=cmap(float(t) / T))

        # https://stackoverflow.com/questions/43805821/matplotlib-add-colorbar-to-non-mappable-object
        norm = mpl.colors.Normalize(vmin=1, vmax=T)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        colorbar = plt.colorbar(sm,
                                ticks=np.arange(1, T + 1, 5),
                                # boundaries=np.arange(-0.05, T + 0.1, .1)
                                )
        colorbar.set_label('Number of Customers')
        plt.title(fr'Indian Buffet Dish Distribution ($\alpha$={alpha})')
        plt.xlabel(r'Number of Dishes after T Customers')
        plt.ylabel(r'P(Number of Dishes after T Customers)')
        plt.xlim(0., max_val)
        plt.savefig(os.path.join(plot_dir, f'crt_table_distribution_alpha={alpha}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


# alpha = alphas[0]
# max_dishes = int(10 * alpha * np.sum(1 / (1 + np.arange(T))))
# dish_indices = np.arange(max_dishes + 1)
# prev_running_harmonic_sum = alpha / 1
# for t in range(2, T):
#     prev_dish_distribution = scipy.stats.poisson.pmf(
#         dish_indices,
#         mu=prev_running_harmonic_sum)
#     new_running_harmonic_sum = prev_running_harmonic_sum + alpha / t
#     new_dish_distribution = scipy.stats.poisson.pmf(
#         dish_indices,
#         mu=new_running_harmonic_sum)
#     print(10)


# import matplotlib.pyplot as plt
#
# plt.imshow(ibp_samples_by_alpha[10.01][0, :, :50])
# plt.ylabel('Customer Index')
# plt.xlabel('Dish Index')
# plt.show()




#
# table_distributions_by_alpha_by_T = {}
# cmap = plt.get_cmap('jet_r')
# for alpha in alphas:
#     ibp_samples = np.mean(ibp_samples_by_alpha[alpha], axis=0)
#     max_dish_idx = np.max(np.argmin(ibp_samples != 0, axis=1))
#
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
#
#     sns.heatmap(
#         data=ibp_samples[:, :max_dish_idx],
#         cbar_kws=dict(label='P(Dish)'),
#         cmap='jet',
#         # mask=ibp_samples[:, :max_dish_idx] == 0,
#         vmin=0.,
#         vmax=1.,
#         ax=axes[0])
#     sns.heatmap(
#         data=analytical_customer_dishes_by_alpha[alpha][:, :max_dish_idx],
#         cbar_kws=dict(label='P(Dish)'),
#         cmap='jet',
#         # mask=analytical_customer_dishes_by_alpha[alpha][:, :max_dish_idx] == 0,
#         vmin=0.,
#         vmax=1.,
#         ax=axes[1])
#
#     # # https://stackoverflow.com/questions/43805821/matplotlib-add-colorbar-to-non-mappable-object
#     # norm = mpl.colors.Normalize(vmin=1, vmax=T)
#     # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     # # sm.set_array([])
#     # colorbar = plt.colorbar(sm,
#     #                         ticks=np.arange(1, T + 1, 5),
#     #                         # boundaries=np.arange(-0.05, T + 0.1, .1)
#     #                         )
#     # colorbar.set_label('Number of Customers')
#     # plt.title(fr'Chinese Restaurant Table Distribution ($\alpha$={alpha})')
#     axes[0].set_title(rf'Monte Carlo Estimate ($\alpha$={alpha})')
#     axes[0].set_ylabel(r'Customer Index')
#     axes[0].set_xlabel(r'Dish Index')
#     axes[1].set_title(rf'Analytical Prediction ($\alpha$={alpha})')
#     axes[1].set_xlabel(r'Dish Index')
#     plt.savefig(os.path.join(plot_dir, f'empirical_ibp={alpha}.png'),
#                 bbox_inches='tight',
#                 dpi=300)
#     plt.show()
#     plt.close()
