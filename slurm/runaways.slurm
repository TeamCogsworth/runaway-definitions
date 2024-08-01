#!/bin/bash
## Job Name
#SBATCH --job-name=cogsworth
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=3:00:00
#SBATCH -o /mnt/home/twagg/runaways/slurm/logs_%a_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tomwagg@uw.edu
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

python /mnt/home/twagg/runaways/src/generate_runaway_pop.py