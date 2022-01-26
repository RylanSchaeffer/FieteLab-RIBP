#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 2                    # two cores
#SBATCH --mem=64G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)

run_one_results_dir_path=${1}
inference_alg_str=${2}
alpha=${3}
beta=${4}
data_dim=${5}
num_gaussians=${6}

# don't remember what this does
export PYTHONPATH=.

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

# -u flushes output buffer immediately
python -u 02_linear_gaussian/fig6_st1_run_one.py \
--run_one_results_dir="${run_one_results_dir_path}" \
--inference_alg_str="${inference_alg_str}" \
--alpha="${alpha}" \
--beta="${beta}" \
--data_dim="${data_dim}" \
--num_gaussians="${num_gaussians}"
