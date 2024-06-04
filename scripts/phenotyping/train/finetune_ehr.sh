#!/bin/bash
#SBATCH -c 5
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name ft_train_ehr
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/phenotyping/finetune/ft_train_ehr_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/phenotyping/finetune/ft_train_ehr_%j.err

source activate mml-ssl

python /scratch/se1525/mml-ssl/run_gpu.py  \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 300 --batch_size 256 --lr 0.001 --transforms_cxr simclrv2 \
--job_number ${SLURM_JOBID} \
--load_state SIMCLR-7459889_epoch_100 \
--file_name SIMCLR-7459889-E100-FTEHR-${SLURM_JOBID} \
--finetune \
--pretrain_type simclr \
--fusion_type lineareval_ehr \
--mode train \
--save_dir /scratch/se1525/mml-ssl/checkpoints/phenotyping/models \
--tag finetuning_ehr \
