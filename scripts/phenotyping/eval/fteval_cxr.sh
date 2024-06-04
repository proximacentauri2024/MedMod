#!/bin/bash
#SBATCH -n 10
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name ft_eval_cxr
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/phenotyping/finetune/ft_eval_cxr_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/phenotyping/finetune/ft_eval_cxr_%j.err

source activate mml-ssl


python /scratch/se1525/mml-ssl/run_gpu.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 10 --batch_size 256 --lr 0.2 --transforms_cxr simclrv2 \
--job_number ${SLURM_JOBID} \
--load_state SIMCLR-4638812-E186-FTCXR-4702562_epoch_26 \
--pretrain_type simclr \
--finetune \
--fusion_type lineareval_cxr \
--mode eval \
--eval_set val \
--save_dir /scratch/se1525/mml-ssl/checkpoints/phenotyping/models \
--tag ft_eval_cxr \
