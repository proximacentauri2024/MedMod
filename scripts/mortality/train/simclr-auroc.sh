#!/bin/bash
#SBATCH -c 5
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name simclr_train
#SBATCH --output /scratch/fs999/shamoutlab/Farah/contrastive-learning-results/checkpoints/phenotyping/simclr/simclr_train_%j.log
source activate /scratch/fs999/shamoutlab/conda-envs/med_fuse

python /scratch/fs999/shamoutlab/Farah/contrastive-learning-jubail/simclr_val_auroc.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--job_number ${SLURM_JOBID} \
--file_name SIMCLR-${SLURM_JOBID} \
--epochs 300 --transforms_cxr simclrv2 --temperature 0.5 \
--batch_size 512 --lr 0.1 \
--pretrain_type simclr \
--mode train \
--fusion_type None \
--save_dir /scratch/fs999/shamoutlab/Farah/contrastive-learning-results/checkpoints/phenotyping/models \
--tag simclr_train \
