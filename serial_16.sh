#!/bin/bash
#SBATCH -p shakhnovich
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 60
#SBATCH --mem=5000
#SBATCH --mail-type=END
#SBATCH --mail-user=awhitney@college.harvard.edu

source new-modules.sh
module load gcc
module load Anaconda

python driver_timed.py serial_16.conf
