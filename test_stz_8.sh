#!/bin/bash
#SBATCH -p shakhnovich
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 10
#SBATCH --mem=1000

source new-modules.sh
module load Anaconda

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/sw/fasrcsw/apps/Core/Anaconda/1.9.2-fasrc01/x/lib/mpi4py/include/

start_dir=$PWD
mkdir -p /scratch/$USER/$SLURM_JOBID
cp *.py /scratch/$USER/$SLURM_JOBID
cp *.pxd /scratch/$USER/$SLURM_JOBID
cp *.so /scratch/$USER/$SLURM_JOBID
cp test.conf /scratch/$USER/$SLURM_JOBID
cd /scratch/$USER/$SLURM_JOBID

mpiexec -n 8 python driver.py test.conf

cp *.out *.dat $start_dir/
cd $start_dir

rm -rf /scratch/$USER/$SLURM_JOBID
