#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:2
#SBATCH --time=1-23:59:59
#SBATCH --cpus-per-task=18
# Output and error files
#SBATCH -o outlogs/job.%J.out
#SBATCH -e errlogs/job.%J.err
    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate medfuse

CUDA_VISIBLE_DEVICES=0  python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode train \
--epochs 50 --batch_size 128 --lr 0.00053985 \
--vision_num_classes 1 --num_classes 10 \
--data_pairs partial_ehr \
--fusion_type uni_ehr --task length-of-stay \
--save_dir checkpoints/length-of-stay/uni_ehr_all \
--labels_set length-of-stay