#!/bin/bash
#SBATCH -c 20
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:3

#SBATCH --job-name cxr_train
#SBATCH --output /scratch/se1525/MedFuse/checkpoints/cxr_rad_full/cxr_train_%j.log
source activate /home/se1525/.conda/envs/medfuse
sh ./scripts/radiology/uni_cxr.sh

