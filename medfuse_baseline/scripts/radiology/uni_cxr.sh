#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:2
#SBATCH --time=3-23:59:59
#SBATCH --cpus-per-task=18
# Output and error files
#SBATCH -o outlogs/job.%J.out
#SBATCH -e errlogs/job.%J.err
    
# Activating conda
eval "$(conda shell.bash hook)"
conda activate medfuse

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py --dim 256 \
--dropout 0.3 --mode train \
--epochs 100 --pretrained \
--vision-backbone resnet34 --data_pairs radiology \
--batch_size 16 --align 0.0 --labels_set radiology --save_dir checkpoints/medfuse_Results \
--fusion_type uni_cxr --layers 2 --vision_num_classes 14 \
--load_state 'checkpoints/medfuse_Results/last_uni_cxr_0.0001_checkpoint.pth.tar'
