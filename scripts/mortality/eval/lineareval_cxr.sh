#!/bin/bash
#SBATCH -n 16
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name linear_eval_mortality_cxr
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/mortality/lineareval/linear_eval_cxr_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/mortality/lineareval/linear_eval_cxr_%j.err
source activate mml-ssl


python /scratch/se1525/mml-ssl/run_gpu.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 10 --batch_size 256 --lr 0.2 --transforms_cxr simclrv2 \
--job_number ${SLURM_JOBID} \
--task in-hospital-mortality \
--labels_set mortality \
--num_classes 1 \
--load_state SIMCLR-7459889-E100-LCXR-7599311_epoch_299 \
--pretrain_type simclr \
--fusion_type lineareval_cxr \
--mode eval \
--eval_set val \
--save_dir /scratch/se1525/mml-ssl/checkpoints/mortality/models \
--tag linear_eval_cxr_mortality \
