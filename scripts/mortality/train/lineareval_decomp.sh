#!/bin/bash
#SBATCH -c 5
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name linear_train_mortality
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/mortality/lineareval/linear_train_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/mortality/lineareval/linear_train_%j.err

source activate mml-ssl

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python /scratch/se1525/mml-ssl/run_gpu.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 400 --batch_size 256 --lr 0.01 --transforms_cxr simclrv2 \
--job_number ${SLURM_JOBID} \
--task decompensation \
--labels_set decompensation \
--num_classes 1 \
--load_state SIMCLR-5627729_epoch_299 \
--file_name SIMCLR-5627729-E299-LC-${SLURM_JOBID} \
--pretrain_type simclr \
--fusion_type lineareval \
--mode train \
--save_dir /scratch/se1525/mml-ssl/checkpoints/decompensation/models \
--tag linear_train_decompensation  \
