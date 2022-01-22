#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=64G                # RAM
#SBATCH --time=120:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=gkml
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u 02_linear_gaussian/analyze_asymptotics.py        # -u flushes output buffer immediately
