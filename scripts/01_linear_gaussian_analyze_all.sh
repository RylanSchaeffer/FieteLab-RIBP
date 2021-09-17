#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=32G                # RAM
#SBATCH --time=01:00:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u 01_linear_gaussian/analyze_all.py        # -u flushes output buffer immediately
