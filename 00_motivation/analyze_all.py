import os
import pandas as pd

import plot_motivation

plot_dir = '00_motivation/results'


datasets = ['Omniglot', 'MNIST']
for dataset in datasets:

    # First, plot frac_var_explained_by_fixed_num_pcs
    frac_var_vs_fixed_pcs_df = pd.read_csv(
        os.path.join(plot_dir, f'{dataset.lower()}_frac_var_explained_by_fixed_num_pcs.csv'),
        index_col=None)

    plot_motivation.plot_frac_var_explained_by_fixed_num_pcs(
        df=frac_var_vs_fixed_pcs_df,
        dataset=dataset,
        plot_dir=plot_dir)

    # Next, plot _num_pcs_to_explain_by_dataset_size
    num_pcs_to_explain_df = pd.read_csv(
        os.path.join(plot_dir, f'{dataset.lower()}_num_pcs_to_explain_by_dataset_size.csv'),
        index_col=None)

    plot_motivation.plot_num_pcs_to_explain_by_dataset_size(
        df=num_pcs_to_explain_df,
        dataset=dataset,
        plot_dir=plot_dir)
