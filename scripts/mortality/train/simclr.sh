#!/bin/bash
#SBATCH -c 5
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH -q nvidia-xxl
#SBATCH --gres=gpu:a100:1

#SBATCH --job-name simclr_train_mortality
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/mortality/simclr/simclr_train_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/mortality/simclr/simclr_train_%j.err

source activate mml-ssl

python /scratch/se1525/mml-ssl/run_gpu.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--job_number ${SLURM_JOBID} \
--file_name SIMCLR-${SLURM_JOBID} \
--epochs 300 --transforms_cxr simclrv2 --temperature 0.5 \
--batch_size 512 --lr 1 \
--pretrain_type simclr \
--mode train \
--fusion_type None \
--save_dir /scratch/se1525/mml-ssl/checkpoints/mortality/models \
--task in-hospital-mortality \
--tag simclr_train_mortality 
