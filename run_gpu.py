# Copyright 2022 Farah E. Shamout
#
# TODO: licsense
# ==============================================================================
"""This script defines the SimCLR model and performs training and evaluation."""


data_dir = '/scratch/fs999/shamoutlab/data/mimic-iv-extracted/'
img_dir = '/scratch/fs999/shamoutlab/data/physionet.org/files//mimic-cxr-jpg/2.0.0'
code_dir = '/scratch/se1525/mml-ssl'
# task = 'phenptyping'

# Import libraries
import sys
import numpy as np
import argparse
import os
import importlib as imp
import re
from pathlib import Path
import pandas as pd
import neptune.new as neptune
from pathlib import Path
import pdb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# Import Pytorch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision


## Performance metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score
# from sklearn.model_selection import RandomizedSearchCV

# Import custom functions
import parser as par
import data_utils as prep


# Import functions from MedFuse
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets
from datasets.fusion import load_cxr_ehr
from ehr_preprocess import ehr_funcs
from simclr_trainer_gpu import SimCLR, train, test, prepare_data_features#, LogisticRegression, train_logreg
import load_tasks as tasks


import warnings
warnings.filterwarnings("ignore")

def initiate_logger(tags):
    logger = pl_loggers.NeptuneLogger(project="shaza-workspace/mml-ssl",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NDU3ZDlmMi01OGEyLTQzMTAtODJmYS01Mjc5N2U4ZjgyMTAifQ==", tags=tags, log_model_checkpoints=False)
    return logger

seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == '__main__':
    
    # Parse arguments
    parser = par.initiate_parsing()
    args = parser.parse_args()
    job_number = args.job_number
    
    path = Path(args.save_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Set cuda device
    if args.device=='0':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  
    elif args.device=='1':
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'   
    elif args.device=='2':
        device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    elif args.device=='3':
        device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    elif args.device=='4':
        device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    elif args.device=='5':
        device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    elif args.device=='6':
        device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
    elif args.device=='7':
        device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    elif args.device=='8':
        device = 'cuda:8' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'None'
    print('Using {} device...'.format(device))        
   
    # Load datasets and initiate dataloaders
    print('Loading datasets...')
    discretizer, normalizer = ehr_funcs(args)
    ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)
    cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)
    train_dl, val_dl, test_dl = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds)
    print("Length of training dataset: " , len(train_dl.dataset))
    print("Length of validation dataset:" , len(val_dl.dataset))
    print("Length of test dataset: " , len(test_dl.dataset))

    # Store arguments after loading datasets
    os.makedirs(os.path.dirname(f"{args.save_dir}/args/args_{job_number}.txt"), exist_ok=True)
    with open(f"{args.save_dir}/args/args_{job_number}.txt", 'w') as results_file:
        print("Storing arguments...")
        for arg in vars(args): 
            print(f"  {arg:<40}: {getattr(args, arg)}")
            results_file.write(f"  {arg:<40}: {getattr(args, arg)}\n")

    # Initiate logger
    neptune_logger = initiate_logger([args.tag, args.job_number])  
    neptune_logger.experiment["args"] = vars(args)
    
    # Load the model, weights (if any), and freeze layers (if any)
    print("Loading model...")
    if args.pretrain_type == 'simclr':
        model = SimCLR(args, train_dl)

    
    # For efficiency, change data loaders for lineareval (not finetune)
    if ('lineareval' in args.fusion_type) & (not args.finetune):
        print("Processing features for linear evaluation...")
        train_dl = prepare_data_features(device, model, train_dl, args.batch_size, args.fusion_layer, args.fusion_type) 
        print("Length of training dataset: " , len(train_dl.dataset))
        val_dl = prepare_data_features(device, model, val_dl, args.batch_size, args.fusion_layer, args.fusion_type)
        print("Length of validatiom dataset:" , len(val_dl.dataset))
        test_dl = prepare_data_features(device, model, test_dl, args.batch_size, args.fusion_layer, args.fusion_type)
        print("Length of test dataset: " , len(test_dl.dataset))
 
    if args.mode == 'train':
        print('==> training')        
        train(model, args, train_dl, val_dl,
              logger=neptune_logger,
              load_state_prefix=args.load_state_simclr)
        test(model, args, test_dl, logger=neptune_logger)
        
    elif args.mode == 'eval':
        if args.eval_set=='val':
            print('==> evaluating on the val set')
            test_dl=val_dl
        elif args.eval_set=='train':
            print('==> evaluating on the train set')
            test_dl=train_dl
        else:
            print('==> evaluating on the test set')
        test(model, args, test_dl, logger=neptune_logger)
                
    else:
        raise ValueError("Incorrect value for args.mode")
    
    
