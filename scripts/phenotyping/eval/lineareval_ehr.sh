#!/bin/bash
#SBATCH -n 16
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name linear_eval_ehr_phenotyping
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/phenotyping/lineareval/linear_eval_ehr_phenotyping_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/phenotyping/lineareval/linear_eval_ehr_phenotyping_%j.err

source activate mml-ssl

python /scratch/se1525/mml-ssl/run_gpu.py  \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 10 --batch_size 256 --lr 0.2 --transforms_cxr simclrv2 \
--job_number ${SLURM_JOBID} \
--load_state SIMCLR-7459889-E100-LEHR-7599289_epoch_299 \
--pretrain_type simclr \
--fusion_type lineareval_ehr \
--mode eval \
--eval_set test \
--save_dir /scratch/se1525/mml-ssl/checkpoints/phenotyping/models \
--tag linear_eval_ehr_phenotyping \
