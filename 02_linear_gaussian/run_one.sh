#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 2                    # two cores
#SBATCH --mem=64G               # RAM
#SBATCH --time=24:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

run_one_results_dir_path=${1}
inference_alg_str=${2}
alpha=${3}
beta=${4}

# don't remember what this does
export PYTHONPATH=.

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

# -u flushes output buffer immediately
python -u 02_linear_gaussian/run_one.py \
--run_one_results_dir="${run_one_results_dir_path}" \
--inference_alg_str="${inference_alg_str}" \
--alpha="${alpha}" \
--beta="${beta}"
