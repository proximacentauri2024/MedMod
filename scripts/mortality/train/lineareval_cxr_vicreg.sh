#!/bin/bash
#SBATCH -c 5
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1


#SBATCH --job-name linear_train_mortality_cxr_vicreg
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/mortality/lineareval/linear_train_cxr_vicreg_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/mortality/lineareval/linear_train_cxr_vicreg_%j.err

source activate mml-ssl

python /scratch/se1525/mml-ssl/run_gpu.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 300 --batch_size 256 --lr 0.001 --transforms_cxr simclrv2 \
--job_number ${SLURM_JOBID} \
--task in-hospital-mortality \
--labels_set mortality \
--num_classes 1 \
--vicreg \
--load_state VICREG-7458877_epoch_56 \
--file_name VICREG-7458877-E56-LCXR-${SLURM_JOBID} \
--pretrain_type simclr \
--fusion_type lineareval_cxr \
--mode train \
--save_dir /scratch/se1525/mml-ssl/checkpoints/mortality/models \
--tag linear_train_cxr_mortality_vicreg \
