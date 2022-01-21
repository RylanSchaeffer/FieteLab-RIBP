#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=32G               # RAM
#SBATCH --time=06:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=gkml
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u 00_motivation/frac_var_explained_by_fixed_num_pcs.py        # -u flushes output buffer immediately
