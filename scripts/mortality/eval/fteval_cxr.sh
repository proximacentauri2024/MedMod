#!/bin/bash
#SBATCH -n 16
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name ft_eval_cxr
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/mortality/finetune/ft_eval_cxr_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/mortality/finetune/ft_eval_cxr_%j.err

source activate mml-ssl

python /scratch/se1525/mml-ssl/run_gpu.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 10 --batch_size 256 --lr 0.2 --transforms_cxr simclrv2 \
--job_number ${SLURM_JOBID} \
--task in-hospital-mortality \
--labels_set mortality \
--num_classes 1 \
--load_state SIMCLR-7459889-E100-FTCXR-7600148_epoch_91 \
--pretrain_type simclr \
--finetune \
--fusion_type lineareval_cxr \
--mode eval \
--eval_set test \
--save_dir /scratch/se1525/mml-ssl/checkpoints/mortality/models \
--tag ft_eval_cxr \
