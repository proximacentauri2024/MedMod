# Copyright 2022 Farah E. Shamout
#
# TODO: licsense
# ==============================================================================
"""This script defines the SimCLR model and performs training and evaluation."""


data_dir = '/scratch/fs999/shamoutlab/data/mimic-iv-extracted/'
img_dir = '/scratch/fs999/shamoutlab/data/'
code_dir = '/scratch/fs999/shamoutlab/Farah/contrastive-learning-jubail/'
task = 'phenotyping'

# Import libraries
import sys
#sys.path.append(f'{code_dir_medfuse}')
import numpy as np
import argparse
import os
import importlib as imp
import re
from pathlib import Path
import pandas as pd
import neptune.new as neptune
from pathlib import Path
import glob
import time
import pickle as pkl

# ## Visualization
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
# from tqdm.notebook import tqdm
# import matplotlib
# matplotlib.use('Agg')

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

# Import custom functions
import parser as par
import data_utils as prep
# from fusion_trainer_farah import FusionTrainer
# from mmtm_trainer import MMTMTrainer
# from daft_trainer import DAFTTrainer


# Import functions from MedFuse
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets
from datasets.fusion import load_cxr_ehr
from ehr_preprocess import ehr_funcs
from simclr_trainer_gpu import SimCLR, train, test, prepare_data_features#, LogisticRegression, train_logreg
import load_tasks as tasks

#sys.path.append('/home/shamoutlab/.local/bin')

import warnings
warnings.filterwarnings("ignore")

def initiate_logger(tags):
    logger = pl_loggers.NeptuneLogger(project="nyuad-cai/cl-img-ehr",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3NWU2ZThkMy03N2Q5LTQyNDYtOTc4MS04MTZkN2ViMjVjZGIifQ==", tags=tags, log_model_checkpoints=False)
    return logger

seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)


def load_weights(model, path):
        checkpoint = torch.load(path)
        own_state = model.state_dict()
        own_keys = list(model.state_dict().keys())
        checkpoint_keys = list(checkpoint['state_dict'].keys())
        
        #print('Total number of checkpoint params = {}'.format(len(checkpoint_keys)))
        #print('Total number of current model params = {}'.format(len(own_keys)))

        count = 0
        changed = []
        for name in own_keys:
            if name not in checkpoint_keys:
                # double check if name exists in a different format
                for x in checkpoint_keys:
                    if name in x:
                        param=checkpoint['state_dict'][x]
                        if isinstance(param, torch.nn.Parameter):
                            param=param.data
                        own_state[name].copy_(param)
                        count+=1
            else:
                param=checkpoint['state_dict'][name]
                if isinstance(param, torch.nn.Parameter):
                    param=param.data
                own_state[name].copy_(param)
                count+=1
        #print('Total number params loaded for model weights = {}'.format(count))
        
        
        return model
    
    
def best_model(num_epochs, results):
    auroc_train = []
    auprc_train = []

    auroc_val = []
    auprc_val = []
    for i in range(0, num_epochs):
        auroc_train.append(results[i]['train_auroc'])
        auprc_train.append(results[i]['train_auprc'])
    
        auroc_val.append(results[i]['val_auroc'])
        auprc_val.append(results[i]['val_auprc'])
    
    max_val_auroc = max(auroc_val) 
    max_val_auprc = max(auprc_val) 

    print(max_val_auroc)
    print(max_val_auprc)

    max_index = auroc_val.index(max_val_auroc)
    return max_index

if __name__ == '__main__':
    # Measure time
    startTime = time.time()
    
    
    # Parse arguments
    parser = par.initiate_parsing()
    args = parser.parse_args()
    job_number = args.job_number
    
    path = Path(args.save_dir)
    path.mkdir(parents=True, exist_ok=True)
    
#     # Set cuda device
#     if torch.cuda.is_available():
#         device = 'cuda'    
#     else:
#         device = 'cpu'
    device ='cpu'
    print('Using {} device...'.format(device))        
   
    # Load datasets and initiate dataloaders
    print('Loading datasets...')
    discretizer, normalizer = ehr_funcs(args)
    ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)
    cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)
    train_dl, val_dl, test_dl = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds)
    
#     # Store arguments after loading datasets
#     with open(f"{args.save_dir}/args/args_{job_number}.txt", 'w') as results_file:
#         print("Storing arguments...")
#         for arg in vars(args): 
#             print(f"  {arg:<40}: {getattr(args, arg)}")
#             results_file.write(f"  {arg:<40}: {getattr(args, arg)}\n")

    ## Get model paths
    load_dir_simclr = args.save_dir
    if 'mortality' in args.save_dir:
        load_dir_simclr = load_dir_simclr.replace('mortality', 'phenotyping')
    
    print(load_dir_simclr+"/{}/*".format(args.load_state))
    paths_models = glob.glob(load_dir_simclr+"/{}/*".format(args.load_state))
    print('\nNumber of epochs =',len(paths_models))
    num_epochs = len(paths_models)
    max_epoch = num_epochs-1
    print('Max epoch idx =', max_epoch)
    
    # Create a directory to save results of current training settings
    args.file_name = args.file_name + '-lr{}-e{}-bs{}'.format(args.lr, args.epochs, args.batch_size)
    
    results_path = args.save_dir+'/'+args.file_name+'/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results_file_path = results_path+'results_epoch_{}.csv'.format(args.eval_epoch) 
    
    print(results_file_path)
    
    if os.path.isfile(results_file_path):
        saved_results = pd.read_csv(results_file_path)
        print(saved_results.head())
        print("Epoch already evaluated for the same settings =", saved_results)
    else:
        print("No results for this epoch yet")
        i = args.eval_epoch
    
        results = {}
        load_state=args.load_state

        print("------------------------------------------------------")
        print("Epoch num=", i)
        print("------------------------------------------------------")


        # Load the model weights and freeze encoders
        print("Loading model...")
        if i <10:
            epoch_num = '0'+str(i)
        else:
            epoch_num = str(i)

        args.load_state = load_state+'_epoch_'+epoch_num
        if args.pretrain_type == 'simclr':
            model = SimCLR(args, train_dl)


        if ('lineareval' in args.fusion_type) & (not args.finetune):
            print("Processing features for linear evaluation...")
            startTime_data = time.time()
            train_dl_pr = prepare_data_features(device, model, train_dl, args.batch_size, args.fusion_layer, args.fusion_type) 
            val_dl_pr = prepare_data_features(device, model, val_dl, args.batch_size, args.fusion_layer, args.fusion_type)
            test_dl_pr = prepare_data_features(device, model, test_dl, args.batch_size, args.fusion_layer, args.fusion_type)
            
            executionTime_data = (time.time() - startTime_data)
            print('Execution time in minutes for processing the data features: ' + str(executionTime_data/60))
            
        # Delete parts of the model for memory purposes
        del model.model.cxr_model
        del model.model.cxr_model_g
        del model.model.ehr_model
        del model.model.ehr_model_g

        # [1] Train LR model and store best checkpoint of the LR model based on validation AUROC 
        print('==> training')        
        print(len(train_dl))
        trainer = train(model, args, train_dl_pr, val_dl_pr, load_state_prefix=args.load_state_simclr)

        print("Best model score = ", trainer.checkpoint_callback.best_model_score)
        print("Best model path = ", trainer.checkpoint_callback.best_model_path)

        # [2] Load best check point and store its training set and validation set AUROC for the respective SIMCLR epoch
        model = load_weights(model, trainer.checkpoint_callback.best_model_path)

        results[i] = {}

        trainer.test(model, train_dl_pr)
        results[i]['train_auroc'] = trainer.logged_metrics['test_auroc'].item()
        results[i]['train_auprc'] = trainer.logged_metrics['test_auprc'].item()

        trainer.test(model, val_dl_pr)
        results[i]['val_auroc'] = trainer.logged_metrics['test_auroc'].item()
        results[i]['val_auprc'] = trainer.logged_metrics['test_auprc'].item()

        trainer.test(model, test_dl_pr)
        results[i]['test_auroc'] = trainer.logged_metrics['test_auroc'].item()
        results[i]['test_auprc'] = trainer.logged_metrics['test_auprc'].item()

        # After evaluation is completed, free up the memory
        del model

   
        new_results = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index':'epoch'})
        saved_results = new_results
        
        
        saved_results.to_csv(results_file_path)
            
    executionTime = (time.time() - startTime)
    print('Execution time in minutes: ' + str(executionTime/60))
