#!/bin/bash
#SBATCH -c 16
#SBATCH -t 96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name medclr_train
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/phenotyping/medclr/medclr_train_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/phenotyping/medclr/medlcr_train_%j.err

source activate mml-ssl

python /scratch/se1525/mml-ssl/run_gpu_medclr.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--job_number ${SLURM_JOBID} \
--file_name MEDCLR-${SLURM_JOBID} \
--epochs 300 --transforms_cxr simclrv2 --temperature 0.01 \
--batch_size 256 --lr 0.1 \
--num_gpu 1 \
--beta_infonce \
--pretrain_type simclr \
--mode train \
--fusion_type None \
--save_dir /scratch/se1525/mml-ssl/checkpoints/phenotyping/models \
--tag medclr_train \
