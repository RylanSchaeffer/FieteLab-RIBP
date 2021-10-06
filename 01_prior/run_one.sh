#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 2                    # two cores
#SBATCH --mem=24G               # RAM
#SBATCH --time=2:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

run_one_results_dir_path=${1}
num_customer=${2}
num_mc_sample=${3}
alpha=${4}
beta=${5}

# don't remember what this does
export PYTHONPATH=.

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

# -u flushes output buffer immediately
python -u 01_prior/run_one.py \
--results_dir_path="${run_one_results_dir_path}" \
--num_customer="${num_customer}" \
--num_mc_sample="${num_mc_sample}" \
--alpha="${alpha}" \
--beta="${beta}"
