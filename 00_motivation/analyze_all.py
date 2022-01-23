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
    plt.show()
    plt.close()
