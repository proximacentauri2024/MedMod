#!/bin/bash
#SBATCH -c 5
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name linear_train
#SBATCH --output /scratch/fs999/shamoutlab/Farah/contrastive-learning-results/checkpoints/mortality/model_selection/model_selection_%j.log
source activate /scratch/fs999/shamoutlab/conda-envs/med_fuse


python /scratch/fs999/shamoutlab/Farah/contrastive-learning-jubail/model_selection.py \
--task in-hospital-mortality \
--labels_set mortality \
--num_classes 1 \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 100 --batch_size 512 --lr 0.01 \
--job_number ${SLURM_JOBID} \
--load_state SIMCLR-954015 \
--width 3 \
--file_name SIMCLR-954015-select-LC \
--pretrain_type simclr \
--fusion_type lineareval \
--mode train \
--fusion_layer 0 \
--save_dir /scratch/fs999/shamoutlab/Farah/contrastive-learning-results/checkpoints/mortality/models \
--tag model_selection \
