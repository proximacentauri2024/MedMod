#!/bin/bash
#SBATCH -n 16
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name linear_eval_mortality
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/mortality/lineareval/linear_eval_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/mortality/lineareval/linear_eval_%j.err

source activate mml-ssl

python /scratch/se1525/mml-ssl/run_gpu.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 10 --batch_size 256 --lr 0.2 --transforms_cxr simclrv2 \
--job_number ${SLURM_JOBID} \
--task in-hospital-mortality \
--labels_set mortality \
--num_classes 1 \
--load_state  MEDCLR-7621341-E28-LC-7626234_epoch_24 \
--pretrain_type simclr \
--fusion_type lineareval \
--mode eval \
--eval_set test \
--save_dir /scratch/se1525/mml-ssl/checkpoints/mortality/models \
--tag linear_eval_mortality \

