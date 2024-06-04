#!/bin/bash
#SBATCH -c 20
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:3

#SBATCH --job-name mortality_eval
#SBATCH --output /scratch/se1525/MedFuse/checkpoints/mortality/medFuse/mortality_eval_%j.log
source activate /home/se1525/.conda/envs/medfuse
sh ./scripts/mortality/eval/medFuse.sh

