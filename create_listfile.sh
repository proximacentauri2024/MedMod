#!/bin/bash
#SBATCH -c 10
#SBATCH -t 48:00:00

#SBATCH --job-name decomp_train_listfile
#SBATCH --output /scratch/se1525/mml-ssl/create_listfile_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/create_listfile_%j.err

source activate mml-ssl

python /scratch/se1525/mml-ssl/create_new_listfile.py
