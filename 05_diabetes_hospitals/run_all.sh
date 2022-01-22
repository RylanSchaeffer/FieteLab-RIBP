#!/bin/bash
#SBATCH -p use-everything
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=01:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u 05_diabetes_hospitals/run_all.py        # -u flushes output buffer immediately
