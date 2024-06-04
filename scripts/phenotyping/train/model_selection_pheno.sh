#!/bin/bash
#SBATCH -c 10
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name linear_train
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/phenotyping/model_selection/model_selection_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/phenotyping/model_selection/model_selection_%j.err 

source activate mml-ssl

python /scratch/se1525/mml-ssl/model_selection.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 180 --batch_size 256 --lr 0.01 \
--job_number ${SLURM_JOBID} \
--load_state SIMCLR-4638812 \
--file_name SIMCLR-4638812-select-LC \
--width 1 \
--task phenotyping \
--labels_set pheno \
--pretrain_type simclr \
--fusion_type lineareval \
--vicreg \
--mode train \
--fusion_layer 0 \
--save_dir /scratch/se1525/mml-ssl/checkpoints/phenotyping/models \
--tag model_selection \
