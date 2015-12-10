#!/bin/bash
#SBATCH -p shakhnovich
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 480
#SBATCH --mem=5000
#SBATCH --mail-type=END
#SBATCH --mail-user=awhitney@college.harvard.edu

source new-modules.sh
module load gcc
module load Anaconda

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/sw/fasrcsw/apps/Core/Anaconda/1.9.2-fasrc01/x/lib/mpi4py/include/

#start_dir=$PWD
#mkdir -p /scratch/$USER/$SLURM_JOBID
#cp *.py /scratch/$USER/$SLURM_JOBID
#cp *.pxd /scratch/$USER/$SLURM_JOBID
#cp *.so /scratch/$USER/$SLURM_JOBID
#cp confs/1_256.conf /scratch/$USER/$SLURM_JOBID
#cd /scratch/$USER/$SLURM_JOBID

mpiexec -n 1 python driver_timed.py confs/1_256.conf

#cp *.out *.dat $start_dir/
#cd $start_dir

#rm -rf /scratch/$USER/$SLURM_JOBID
