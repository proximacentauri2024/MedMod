#!/bin/bash
#SBATCH -c 20
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name los_train
#SBATCH --output /scratch/se1525/MedFuse/checkpoints/length-of-stay/uni_ehr_all/los_train_%j.log
source activate /home/se1525/.conda/envs/medfuse
sh ./scripts/length-of-stay/train/uni_all.sh

