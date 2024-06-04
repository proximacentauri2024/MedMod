# Copyright 2022 Farah E. Shamout
#
# TODO: licsense
# ==============================================================================
"""This script defines the different fusion functions that can be used with SimCLR and baseline models."""

#TODO: move to models folder later

import os

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

# Pytorch flask to get LARS
import flash
from flash.core.optimizers import LARS

## Performance metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# Other
from copy import deepcopy
from tqdm import tqdm

# Custom
from encoders import LSTM, CXRModels
import load_tasks as tasks


class Fusion(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_dim=args.hidden_dim

        fusion_dim = {0:{},
                      3:{}}

        if 'ehr' not in args.fusion_type:
            # Base model chest X-ray modality f(.)
            self.cxr_model = CXRModels(args, args.hidden_dim)
            # MLP for chest X-ray modality g(.)
            w=args.width
            self.cxr_model_g = nn.Sequential(
                self.cxr_model.vision_backbone.fc,  # Linear(ResNet output, 4*hidden_dim)
                nn.Linear(512, w*hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(w*hidden_dim, w*hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(w*hidden_dim, w*hidden_dim, bias=False)
            )

            fusion_dim[0]['cxr']=self.cxr_model.feats_dim
            fusion_dim[3]['cxr']=w*hidden_dim

        if 'cxr' not in args.fusion_type:
            # Base model EHR f(.)
            w=args.width
            self.ehr_model = LSTM(hidden_dim=args.hidden_dim, input_dim=76, num_classes=w*args.hidden_dim, dropout=args.dropout, layers=args.layers)

            # MLP for EHR modality g(.) 
            self.ehr_model_g = nn.Sequential(
                self.ehr_model.dense_layer,  # this is identify in encoders.py
                nn.Linear(128, w*hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(w*hidden_dim, w*hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(w*hidden_dim, w*hidden_dim, bias=False)
            )

            fusion_dim[0]['ehr']=self.ehr_model.feats_dim
            fusion_dim[3]['ehr']=w*hidden_dim
        
        
        # Single layer for linear evaluation of representations
        if self.args.fusion_type == 'lineareval_ehr':
            feats_dim = fusion_dim[args.fusion_layer]['ehr']
            #feats_dim = self.ehr_model.feats_dim
        
        elif self.args.fusion_type == 'lineareval_cxr':
            feats_dim = fusion_dim[args.fusion_layer]['cxr']
            #feats_dim = self.cxr_model.feats_dim
        
        else:
            feats_dim = fusion_dim[args.fusion_layer]['ehr'] + fusion_dim[args.fusion_layer]['cxr']
            #feats_dim = self.ehr_model.feats_dim + self.cxr_model.feats_dim
        
        # print(self.args.fusion_type)
        if self.args.fusion_type != 'None':
            self.fused_cls = nn.Sequential(
                nn.Linear(feats_dim, self.args.num_classes),
                nn.Sigmoid()
            )
        
    def forward(self, x=None, seq_lengths=None, img=None, pairs=None):
        # New for SimCLR
        if self.args.fusion_type == 'lineareval_ehr':
            return self.forward_uni_eval_ehr(x, seq_lengths=seq_lengths)
        elif self.args.fusion_type == 'lineareval_cxr':
            return self.forward_uni_eval_cxr(img=img)
        elif self.args.fusion_type in ['joint',  'early', 'late_avg', 'unified', 'lineareval']:
            return self.forward_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs)
        else:
            return self.forward_simclr(x, seq_lengths=seq_lengths, img=img, pairs=pairs)
        
    def forward_simclr(self, x, seq_lengths, img, pairs=None):
        if self.args.mode == 'eval':
            feats_img_0 = self.cxr_model(img)
            feats_img_3 = self.cxr_model_g(feats_img_0)
            feats_ehr_0 = self.ehr_model(x, seq_lengths)
            feats_ehr_3 = self.ehr_model_g(feats_ehr_0)
            
            return feats_ehr_0, feats_ehr_3, feats_img_0, feats_img_3
            
        else:
            feats_img = self.cxr_model(img)
            feats_img = self.cxr_model_g(feats_img)
            feats_ehr = self.ehr_model(x, seq_lengths)
            feats_ehr = self.ehr_model_g(feats_ehr)
        
            return feats_ehr, feats_img
    
    def forward_fused(self, x, seq_lengths=None, img=None, pairs=None ):
        if ('lineareval' in self.args.fusion_type) & (not self.args.finetune):
            ehr_feats = x
            cxr_feats = img
        else:
            ehr_feats = self.ehr_model(x, seq_lengths) #ehr_preds , 
            cxr_feats = self.cxr_model(img) #cxr_preds, _ , 
#         projected = self.projection(cxr_feats)

        feats = torch.cat([ehr_feats, cxr_feats], dim=1)
        fused_preds = self.fused_cls(feats)

        return {
            'early': fused_preds, 
            'joint': fused_preds, 
            'lineareval': fused_preds,
            'ehr_feats': ehr_feats,
#             'cxr_feats': projected,
            'unified': fused_preds
            }
    
    def forward_uni_eval_cxr(self, img ):
        if ('lineareval' in self.args.fusion_type) & (not self.args.finetune):
            cxr_feats = img
        else:
            cxr_feats = self.cxr_model(img)
        preds = self.fused_cls(cxr_feats)
        return {
            'lineareval_cxr': preds,
            }
    
    def forward_uni_eval_ehr(self, x, seq_lengths=None):
        if ('lineareval' in self.args.fusion_type) & (not self.args.finetune):
            ehr_feats = x
        else:
            ehr_feats = self.ehr_model(x, seq_lengths)
        preds = self.fused_cls(ehr_feats)
        return {
            'lineareval_ehr': preds,
            }     
            