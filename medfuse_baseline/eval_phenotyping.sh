#!/bin/bash
#SBATCH -c 20
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:3

#SBATCH --job-name phenotyping_eval
#SBATCH --output /scratch/se1525/MedFuse/checkpoints/phenotyping/medFuse/phenptyping_eval_%j.log
source activate /home/se1525/.conda/envs/medfuse
sh ./scripts/phenotyping/eval/medFuse.sh

