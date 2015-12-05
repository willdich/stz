#!/bin/bash
#SBATCH -p general
#SBATCH -n 8
#SBATCH -N 1-8
#SBATCH -t 10
#SBATCH --mem=500

source new-modules.sh
module load gcc openmpi
module load Anaconda

start_dir=$PWD
mkdir -p /scratch/$USER/$SLURM_JOBID
cp *.py /scratch/$USER/$SLURM_JOBID
cp *.pyx /scratch/$USER/$SLURM_JOBID
cp *.pxd /scratch/$USER/$SLURM_JOBID
cp test.conf /scratch/$USER/$SLURM_JOBID
cd /scratch/$USER/$SLURM_JOBID

mpiexec -n 8 python driver.py test.conf

cp *.out *.dat $start_dir/
cd $start_dir

rm -rf /scratch/$USER/$SLURM_JOBID
