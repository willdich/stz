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

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/sw/fasrcsw/apps/Core/Anaconda/1.9.2-fasrc01/x/lib/mpi4py/include/

python driver_timed.py serial_64.conf

