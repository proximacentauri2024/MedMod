#!/bin/bash
#SBATCH -c 5
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name simclr_eval
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/phenotyping/simclr/simclr_eval_%j.log
source activate mml-ssl

python /scratch/se1525/mml-ssl/run_gpu.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--job_number ${SLURM_JOBID} \
--file_name SIMCLR-4499745-EVAL \
--load_state SIMCLR-4499745_epoch_148 \
--epochs 1 --transforms_cxr simclrv2 --temperature 0.5 \
--batch_size 512 --lr 1 \
--pretrain_type simclr \
--mode eval \
--eval_set val \
--fusion_type None \
--save_dir /scratch/se1525/mml-ssl/checkpoints/phenotyping/models \
--tag simclr_eval \
