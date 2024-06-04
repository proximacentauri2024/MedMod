#!/bin/bash
#SBATCH -c 20
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:3

#SBATCH --job-name phenotype_train
#SBATCH --output /scratch/se1525/MedFuse/checkpoints/phenotyping/uni_ehr_full/phenotyping_train_%j.log
source activate /home/se1525/.conda/envs/medfuse
sh ./scripts/phenotyping/train/uni_all.sh

