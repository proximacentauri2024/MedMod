#!/bin/bash
#SBATCH -c 5
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name simclr_eval
#SBATCH --output /scratch/fs999/shamoutlab/Farah/contrastive-learning-results/checkpoints/mortality/simclr/simclr_eval_%j.log
source activate /scratch/fs999/shamoutlab/conda-envs/med_fuse

python /scratch/fs999/shamoutlab/Farah/contrastive-learning-jubail/run_gpu.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--task in-hospital-mortality \
--labels_set mortality \
--num_classes 1 \
--job_number ${SLURM_JOBID} \
--file_name SIMCLR-760276-EVAL \
--load_state SIMCLR-760276_epoch_148 \
--epochs 1 --transforms_cxr simclrv2 --temperature 0.5 \
--batch_size 512 --lr 1 \
--pretrain_type simclr \
--mode eval \
--eval_set val \
--fusion_type None \
--save_dir /scratch/fs999/shamoutlab/Farah/contrastive-learning-results/checkpoints/mortality/models \
--tag simclr_eval \
