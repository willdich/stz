#!/bin/bash
#SBATCH -p general
#SBATCH -n 16
#SBATCH -N 1-8
#SBATCH -t 10
#SBATCH --mem=100

source new-modules.sh
module load gcc openmpi
module load Anaconda

start_dir=$PWD
mkdir -p /scratch/$USER/$SLURM_JOBID
cp driver.py /scratch/$USER/$SLURM_JOBID
cp test.conf /scratch/$USER/$SLURM_JOBID
cd /scratch/$USER/$SLURM_JOBID

mpiexec -n 16 python driver.py test.conf

cp  * $start_dir/
cd $start_dir

rm -rf /scratch/$USER/$SLURM_JOBID
