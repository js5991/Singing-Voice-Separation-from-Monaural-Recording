#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=15:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=opt_rpca
#SBATCH --mail-type=END
##SBATCH --mail-user=js5991@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python/intel/2.7.12
module load librosa/intel/0.5.0

python Python-rPCA_implementation.py
