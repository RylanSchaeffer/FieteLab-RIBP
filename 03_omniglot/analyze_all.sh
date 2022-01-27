#!/bin/bash
#SBATCH -p use-everything
#SBATCH -n 4                    # two cores
#SBATCH --mem=64G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u 03_omniglot/analyze_all.py        # -u flushes output buffer immediately
