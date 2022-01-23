import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


# Set style
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 16
sns.set_style("whitegrid")

plot_dir = '00_motivation/results'


datasets = ['Omniglot']  # , 'MNIST'
for dataset in datasets:

    # First, plot frac_var_explained_by_fixed_num_pcs
    df = pd.read_csv(
        os.path.join(plot_dir, f'{dataset.lower()}_frac_var_explained_by_fixed_num_pcs.csv'),
        index_col=None)

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

    # Next, plot _num_pcs_to_explain_by_dataset_size
    df = pd.read_csv(
        os.path.join(plot_dir, f'{dataset.lower()}_num_pcs_to_explain_by_dataset_size.csv'),
        index_col=None)
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


