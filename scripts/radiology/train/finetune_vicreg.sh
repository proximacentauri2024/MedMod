#!/bin/bash
#SBATCH -c 5
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1

#SBATCH --job-name ft_train
#SBATCH --output /scratch/se1525/mml-ssl/checkpoints/radiology/finetune/ft_train_vicreg_%j.log
#SBATCH -e /scratch/se1525/mml-ssl/checkpoints/radiology/finetune/ft_train_vicreg_%j.err

source activate mml-ssl

python /scratch/se1525/mml-ssl/run_gpu.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--epochs 300 --batch_size 256 --lr 0.001 --transforms_cxr simclrv2 \
--job_number ${SLURM_JOBID} \
--labels_set radiology \
--num_classes 14 \
--retrieve_cxr all \
--vicreg \
--load_state VICREG-7458877_epoch_56 \
--file_name VICREG-7458877-E56-FT-${SLURM_JOBID} \
--finetune \
--pretrain_type simclr \
--fusion_type lineareval \
--mode train \
--save_dir /scratch/se1525/mml-ssl/checkpoints/radiology/models \
--tag finetuning_train_vicreg_radiology \
