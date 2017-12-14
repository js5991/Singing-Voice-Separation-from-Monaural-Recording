#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=lstm
#SBATCH --mail-type=END
##SBATCH --mail-user=js5991@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:2

module purge
module load python/intel/2.7.12
module load pytorch/python2.7/0.3.0_4
module load librosa/intel/0.5.0


module load cuda/8.0.44
module load cudnn/8.0v5.1

python LSTM.py

