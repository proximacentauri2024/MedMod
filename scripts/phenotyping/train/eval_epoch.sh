#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -n 8


#SBATCH --job-name linear_train
#SBATCH --output /scratch/fs999/shamoutlab/Farah/contrastive-learning-results/checkpoints/phenotyping/model_selection/epoch_eval_%j.log
source activate /scratch/fs999/shamoutlab/conda-envs/med_fuse


python /scratch/fs999/shamoutlab/Farah/contrastive-learning-jubail/epoch_evaluation.py \
--device cpu \
--vision-backbone resnet34 \
--epochs 100 --batch_size 512 --lr 0.01 \
--job_number ${SLURM_JOBID} \
--load_state VICREG-1556899 \
--file_name VICREG-1556899-select-LC \
--vicreg \
--eval_epoch 0 \
--width 3 \
--pretrain_type simclr \
--fusion_type lineareval \
--mode train \
--fusion_layer 0 \
--save_dir /scratch/fs999/shamoutlab/Farah/contrastive-learning-results/checkpoints/phenotyping/models \
--tag eval_epoch \
