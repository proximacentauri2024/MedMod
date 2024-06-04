#!/bin/bash
#SBATCH -c 5
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name simclr_train
#SBATCH --output /scratch/fs999/shamoutlab/Farah/contrastive-learning-results/checkpoints/phenotyping/time_simclr/time_simclr_train_%j.log
source activate /scratch/fs999/shamoutlab/conda-envs/med_fuse

python /scratch/fs999/shamoutlab/Farah/contrastive-learning-jubail/run_gpu.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--job_number ${SLURM_JOBID} \
--file_name T-SIMCLR-${SLURM_JOBID} \
--epochs 300 --transforms_cxr simclrv2 --temperature 0.01 \
--beta_infonce \
--batch_size 64 --lr 0.5 \
--pretrain_type simclr \
--width 3 \
--mode train \
--fusion_type None \
--save_dir /scratch/fs999/shamoutlab/Farah/contrastive-learning-results/checkpoints/phenotyping/models \
--tag time_simclr_train \
