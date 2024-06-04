#!/bin/bash
#SBATCH -c 20
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:3

#SBATCH --job-name mortality_train
#SBATCH --output /scratch/se1525/MedFuse/checkpoints/mortality/uni_ehr_all/mortality_train_%j.log
source activate /home/se1525/.conda/envs/medfuse
sh ./scripts/mortality/train/uni_all.sh

