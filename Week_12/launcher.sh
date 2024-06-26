#!/usr/bin/env bash

##

# Execution script EML 13.1/13.2.

##

#SBATCH --job-name=pytorch_distributed

#SBATCH --output=pytorch_distributed_%j.out

#SBATCH -p short

#SBATCH -N 1

#SBATCH --cpus-per-task=96

#SBATCH --time=01:00:00

#SBATCH --mail-type=all

#SBATCH --mail-user=david.baier@uni-jena.de


echo "submit host:"

echo $SLURM_SUBMIT_HOST

echo "submit dir:"

echo $SLURM_SUBMIT_DIR

echo "nodelist:"

echo $SLURM_JOB_NODELIST


# activate conda environment

module load tools/anaconda3/2021.05

source "$(conda info -a | grep CONDA_ROOT | awk -F ' ' '{print $2}')"/etc/profile.d/conda.sh

conda activate /work/EML/pytorch_env_new


# train MLP

#cd $HOME/~/EML_2024/Week_12

export PYTHONUNBUFFERED=TRUE

python main.py