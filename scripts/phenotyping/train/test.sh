#!/bin/bash
#SBATCH -c 5
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1

source activate /scratch/fs999/shamoutlab/conda-envs/med_fuse


python test.py
