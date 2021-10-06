#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=32G               # RAM
#SBATCH --time=06:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u 00_motivation/num_pcs_to_explain_threshold_by_dataset_size.py        # -u flushes output buffer immediately
