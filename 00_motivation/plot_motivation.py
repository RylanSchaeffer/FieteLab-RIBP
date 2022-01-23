import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16
sns.set_style("whitegrid")


def plot_frac_var_explained_by_fixed_num_pcs(df: pd.DataFrame,
                                             dataset: str,
                                             plot_dir: str):
    sns.lineplot(data=df,
                 x='dataset_size',
                 y='frac_var_explained',
                 hue='Num PCs',
                 legend='full',  # necessary to force seaborn to not try binning based on hue
                 )
    plt.title(f'{dataset}')
    plt.xlabel('Dataset Size')
    plt.ylabel(f'Fraction of Variance Explained')
    plt.grid(visible=True, axis='both')
    plt.savefig(os.path.join(plot_dir,
                             f'{dataset.lower()}_frac_var_explained_by_fixed_num_pcs.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_num_pcs_to_explain_by_dataset_size(df: pd.DataFrame,
                                            dataset: str,
                                            plot_dir: str):
    g = sns.lineplot(data=df,
                     x='dataset_size',
                     y='num_pcs',
                     hue='threshold',
                     legend='full',  # necessary to force seaborn to not try binning based on hue
                     )
    plt.title(f'{dataset}')
    plt.xlabel('Dataset Size')
    plt.ylabel(f'Num PCs to Explain % Variance')
    plt.grid(visible=True, axis='both')
    legend = g.legend()
    legend.texts[0].set_text("% Variance")
    plt.savefig(os.path.join(plot_dir,
                             f'{dataset.lower()}_num_pcs_to_explain_by_dataset_size.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
