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

# choose velocity dispersion from array based on slurm task id
disp_array=(0.1 2)
disp=${disp_array[$SLURM_ARRAY_TASK_ID]}

output_path=/mnt/home/twagg/ceph/pops/runaways/clustered/disp_${disp}_r_1.h5
python /mnt/home/twagg/runaways/src/generate_runaway_pop.py -d $disp -o $output_path -r 1
