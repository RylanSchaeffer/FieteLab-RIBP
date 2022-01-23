import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16  # was previously 22
sns.set_style("whitegrid")


def plot_neg_log_posterior_predictive_by_alpha_beta(
        inf_algorithms_results_df: pd.DataFrame,
        plot_dir: str):
    # sns.scatterplot(
    #     data=inf_algorithms_results_df,
    #     x='alpha',
    #     y='beta',
    #     hue='negative_log_posterior_predictive',
    #     legend="full",
    # )

    sc = plt.scatter(
        x=inf_algorithms_results_df['alpha'],
        y=inf_algorithms_results_df['beta'],
        c=inf_algorithms_results_df['negative_log_posterior_predictive'],
    )
    plt.colorbar(sc)
    plt.grid(visible=True, axis='both')
    plt.savefig(os.path.join(plot_dir,
                             'negative_log_posterior_predictive_by_alpha_beta.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
