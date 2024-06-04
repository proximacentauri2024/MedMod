
# Import Pytorch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
import math

# Import other useful libraries
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier
import pickle
from flash.core.optimizers import LARS
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

# Import custom libraries/functions
from encoders import LSTM, CXRModels
import load_tasks as tasks
from fusion_models import Fusion
from utils import get_model_performance, get_bin_custom, CustomBins, mean_absolute_percentage_error


gpu_num=os.environ['CUDA_VISIBLE_DEVICES']

# Set cuda device
if gpu_num=='0':
    gpu=[0]
elif gpu_num=='1':
    gpu=[1]
elif gpu_num=='2':
    gpu=[2]
elif gpu_num=='3':
    gpu=[3]
elif gpu_num=='4':
    gpu=[4]
elif gpu_num=='5':
    gpu=[5]
elif gpu_num=='6':
    gpu=[6]
elif gpu_num=='7':
    gpu=[7]
elif gpu_num=='8':
    gpu=[8]
else:
    gpu=['None']
print('Using {} device...'.format(gpu)) 
       
class SimCLR(pl.LightningModule):

    def __init__(self, args, train_dl):

        super().__init__()
        assert args.temperature > 0.0, 'The temperature must be a positive float!'
        self.warmup_epochs= 10 #int(0.05*max_epochs) (10 as in SimCLR)
        self.automatic_optimization = False
        
        self.num_train_batches=len(train_dl)
        self.batch_size=args.batch_size
        hidden_dim=args.hidden_dim
        self.args=args
        self.LABEL_COLUMNS = tasks.load_labels(args.task, args.labels_set)
        self.task = args.task
        
        # Load the architecture based on args
        self.model = Fusion(args)
        self.load_weights()
        self.freeze_weights()
        
        
    def load_weights(self):
        # loads both encoders for simclr
        load_dir_simclr = self.args.save_dir

        if self.args.load_state is not None: 
            if 'LC' not in self.args.load_state and 'LEHR' not in self.args.load_state and 'LCXR' not in self.args.load_state and self.args.mode=='train' :
                if 'mortality' in self.args.save_dir:
                    load_dir_simclr = load_dir_simclr.replace('mortality', 'phenotyping')
                elif 'decompensation' in self.args.save_dir:
                    load_dir_simclr = load_dir_simclr.replace('decompensation', 'phenotyping')
                elif 'length-of-stay' in self.args.save_dir:
                    load_dir_simclr = load_dir_simclr.replace('length-of-stay', 'phenotyping')
                elif 'readmission' in self.args.save_dir:
                    load_dir_simclr = load_dir_simclr.replace('readmission', 'phenotyping')
                elif 'radiology' in self.args.save_dir:
                    load_dir_simclr = load_dir_simclr.replace('radiology', 'phenotyping')
                    
            if 'epoch' in self.args.load_state:
                model_dir='/'+self.args.load_state.split('_epoch')[0] + '/'
                if self.args.tag == 'eval_epoch':
                    checkpoint = torch.load(load_dir_simclr + model_dir + self.args.load_state+".ckpt", map_location="cpu")
                else:
                    checkpoint = torch.load(load_dir_simclr + model_dir + self.args.load_state+".ckpt")    
            else:
                if self.args.tag == 'eval_epoch':
                    checkpoint = torch.load(os.path.join(load_dir_simclr, self.args.load_state+".ckpt"), map_location="cpu")
                else:
                    checkpoint = torch.load(os.path.join(load_dir_simclr, self.args.load_state+".ckpt"))
            own_state = self.model.state_dict()
            own_keys = list(own_state.keys())
            checkpoint_keys = list(checkpoint['state_dict'].keys())
            
            print('Total number of checkpoint params = {}'.format(len(checkpoint_keys)))
            print('Total number of current model params = {}'.format(len(own_keys)))

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
                            # print("Name", name, "Size of own_state[name]:", own_state[name].size())
                            # print("x", x , "Size of param:", param.size())
                            own_state[name].copy_(param)
                            count+=1
                else:
                    param=checkpoint['state_dict'][name]
                    if isinstance(param, torch.nn.Parameter):
                        param=param.data
                    own_state[name].copy_(param)
                    count+=1
            print('Total number params loaded for model weights = {}'.format(count))
        
    def freeze_weights(self):
        if self.args.finetune:
            if 'ehr' not in self.args.fusion_type:
                print("freezing cxr projection head")
                self.freeze(self.model.cxr_model_g)
            if 'cxr' not in self.args.fusion_type: 
                print("freezing ehr projection head")   
                self.freeze(self.model.ehr_model_g)
        else: 
            if 'lineareval' in self.args.fusion_type:
                print('freezing encoders')
                if 'ehr' not in self.args.fusion_type:
                    self.freeze(self.model.cxr_model)
                    self.freeze(self.model.cxr_model_g)
                if 'cxr' not in self.args.fusion_type:
                    self.freeze(self.model.ehr_model)
                    self.freeze(self.model.ehr_model_g) 

    def freeze(self, model):
        for p in model.parameters():
            p.requires_grad = False     
    
    def configure_optimizers(self):
        
        if self.args.fusion_type == 'None':
            # Scaled learning rate in case of multiple GPUs
            if self.args.num_gpu > 1:
                effective_batchsize = self.args.batch_size*self.args.num_gpu
                scaled_lr = self.args.lr*effective_batchsize/self.args.batch_size
            else:
                scaled_lr = self.args.lr 
                        
            # Optimizer
            optimizer = LARS(self.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=self.args.weight_decay)
            
            # Note that the order of the below affects the initial starting learning rate, hence do not change.
            # Main scheduler
            mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, verbose=False)
            # Learning rate warmup
            lambda1= lambda epoch : (epoch+1)/self.warmup_epochs
            warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1, verbose=False)
                         
            return [optimizer], [mainscheduler, warmupscheduler]
        
        
        else:
            optimizer_adam = optim.AdamW(self.parameters(), lr=self.args.lr) #, weight_decay=self.args.weight_decay)
            lr_scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer_adam,milestones=[int(self.args.epochs*0.6),
                                                       int(self.args.epochs*0.8)],gamma=0.1)
            return [optimizer_adam], [lr_scheduler_adam]
                
    def logging_status(self, mode):
        if mode == 'train':
            on_step=True
            on_epoch=True
        else:
            on_step=False # Report for the sake of naming but it's not useful
            on_epoch=True
        return on_step, on_epoch
    
    
    def bce_loss(self, preds, y, mode='train'):
        
        if self.args.task=="length-of-stay":
            loss = nn.CrossEntropyLoss()(preds,y)
        else:
            loss = nn.BCELoss()(preds, y)
        
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
            
        on_step=False
        on_epoch=True
        
        return loss 
    
    
    def info_nce_loss(self, feats_ehr, feats_img, mode='train'):
        # Calculate cosine similarity matrix
        cos_sim = F.cosine_similarity(feats_img[:,None,:], feats_ehr[None,:,:], dim=-1)
        cos_sim = cos_sim /  self.args.temperature
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool,  device=cos_sim.device)
        cos_sim_negative = torch.clone(cos_sim)
        cos_sim_negative.masked_fill_(self_mask, -9e15)
        
        # Compute based on img->ehr
        nll_1 = cos_sim[self_mask] - torch.logsumexp(cos_sim_negative, dim=1)
        
        # Compute based on ehr->img
        nll_2 = cos_sim[self_mask] - torch.logsumexp(cos_sim_negative, dim=0) 
        
        # Total loss 
        loss = -(nll_1 + nll_2).mean()
            
        # Logging ranking metrics
        on_step, on_epoch = self.logging_status(mode)
                     
        return loss
    
    
    
    def off_diagonal(self,x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    
    def vicreg_loss(self, feats_ehr, feats_img, mode='train'):
        x = feats_ehr
        y = feats_img
        repr_loss = F.mse_loss(x, y)
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)

        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2 
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
                
        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        
        num_features = len(cov_x) #TODO as arg
                
        cov_loss_x = self.off_diagonal(cov_x).pow_(2).sum().div(num_features)
        cov_loss_y = self.off_diagonal(cov_y).pow_(2).sum().div(num_features)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(num_features) + self.off_diagonal(cov_y).pow_(2).sum().div(num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        on_step, on_epoch = self.logging_status(mode)
        
        return loss, std_loss_x, std_loss_y, cov_loss_x, cov_loss_y, repr_loss
        
        
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        mode = 'train'
        # Forward pass for SimCLR
        if ((self.args.fusion_type=='None') & (self.args.beta_infonce == False) & (self.args.vicreg == False)):
            ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch
            ehr = torch.from_numpy(ehr).float()
            ehr = ehr.to(self.device)
            feats_ehr, feats_img = self.model(ehr, seq_lengths, imgs)
            loss = self.info_nce_loss(feats_ehr, feats_img, mode)
            self.log(mode+'_loss', loss, on_step=True, on_epoch=True) 
            
            
        elif ((self.args.fusion_type=='None') & (self.args.beta_infonce == True) & (self.args.vicreg == False)):
            ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs, time_diff = batch
            ehr = torch.from_numpy(ehr).float()
            ehr = ehr.to(self.device)
            feats_ehr, feats_img = self.model(ehr, seq_lengths, imgs)
            
            # Compute and log modifed infoNCE loss
            k = self.args.k
            loss = self.modified_info_nce_loss(feats_ehr, feats_img, time_diff, k, mode)
            self.log(mode+'_loss', loss, on_step=True, on_epoch=True) 
            
        elif ((self.args.fusion_type=='None') & (self.args.beta_infonce == False) & (self.args.vicreg == True)):
            ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch
            ehr = torch.from_numpy(ehr).float()
            ehr = ehr.to(self.device)
            feats_ehr, feats_img = self.model(ehr, seq_lengths, imgs)
            
            # Compute and log vicreg loss
            loss, std_loss_x, std_loss_y, cov_loss_x, cov_loss_y, repr_loss = self.vicreg_loss(feats_ehr, feats_img, mode)
            self.log(mode+'_std_ehr', std_loss_x, on_step=True, on_epoch=True) 
            self.log(mode+'_std_img', std_loss_y, on_step=True, on_epoch=True)
            self.log(mode+'_cov_ehr', cov_loss_x, on_step=True, on_epoch=True) 
            self.log(mode+'_cov_img', cov_loss_y, on_step=True, on_epoch=True)  
            self.log(mode+'_repr_loss', repr_loss, on_step=True, on_epoch=True)  
            self.log(mode+'_loss', loss, on_step=True, on_epoch=True) #, logger=True)
            
                        
        else:
            if self.args.finetune:
                ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch

                ehr = torch.from_numpy(ehr).float()
                ehr = ehr.to(self.device)
                imgs = imgs.to(self.device)
                output = self.model(x=ehr, seq_lengths=seq_lengths, img=imgs)
                y_ehr = torch.from_numpy(y_ehr)
        
            else: # Features are already processed for linear classifier
                seq_lengths=None
                if 'ehr' in self.args.fusion_type:
                    ehr, y_ehr = batch
                    ehr = ehr.to(self.device)
                    output = self.model(x=ehr,seq_lengths=seq_lengths)
                elif 'cxr' in self.args.fusion_type:
                    imgs, y_cxr, y_ehr = batch
                    imgs = imgs.to(self.device)
                    output = self.model(img=imgs)
                else:
                    ehr, imgs, y_ehr, y_cxr = batch
                    ehr = ehr.to(self.device)
                    imgs = imgs.to(self.device)
                    output = self.model(x=ehr, seq_lengths=seq_lengths, img=imgs)

            if self.args.labels_set=='radiology':
                y = y_cxr.float()
                y = y.to(self.device)
            else:
                y = y_ehr.float()
                y = y.to(self.device)

            preds = output[self.args.fusion_type].squeeze()

            # Compute and log BCE loss
            if self.args.task == "length-of-stay":
                # Assuming the loss function is CrossEntropyLoss 
                y_true_bins = torch.tensor([get_bin_custom(y_item, CustomBins.nbins) for y_item in y.cpu().numpy()], dtype=torch.long).to(self.device)
                loss = self.bce_loss(preds, y_true_bins)
            else:
                loss = self.bce_loss(preds, y, mode)
            self.log(mode+'_loss', loss, on_step=True, on_epoch=True) #, logger=True)
    
        # Backpropagate
        self.manual_backward(loss)
        # Optimizer step
        opt.step()          
        # Learning rate step
        if self.args.fusion_type=='None':
            mainscheduler, warmupscheduler = self.lr_schedulers()
            if (self.trainer.is_last_batch) and (self.trainer.current_epoch < self.warmup_epochs-1):
                warmupscheduler.step()
            elif (self.trainer.is_last_batch) and (self.trainer.current_epoch >= self.warmup_epochs-1):
                mainscheduler.step()
            return {'loss': loss, 'feats_ehr': feats_ehr.detach().cpu(), 'feats_img': feats_img.detach().cpu(), 'y_ehr':y_ehr}
        else:
            return {'loss': loss, 'preds': preds, 'y': y}

    
    def validation_step(self, batch, batch_idx):
        mode='val'
        # Forward pass for SimCLR
        if ((self.args.fusion_type=='None') & (self.args.beta_infonce == False)):
            ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch
            ehr = torch.from_numpy(ehr).float()
            ehr = ehr.to(self.device)
            feats_ehr, feats_img = self.model(ehr, seq_lengths, imgs) 
            
            # Compute and log infoNCE loss
            loss = self.info_nce_loss(feats_ehr, feats_img, mode)
            self.log(mode+'_loss_epoch', loss, on_step=False, on_epoch=True) #, logger=True)
            
            return {'loss': loss, 'feats_ehr': feats_ehr.detach().cpu(), 'feats_img': feats_img.detach().cpu(), 'y_ehr':y_ehr}
        
        elif ((self.args.fusion_type=='None') & (self.args.beta_infonce == True)):
            ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs, time_diff = batch
            ehr = torch.from_numpy(ehr).float()
            ehr = ehr.to(self.device)
            feats_ehr, feats_img = self.model(ehr, seq_lengths, imgs)
            
            # Compute and log infoNCE loss
            k = self.args.k
            loss = self.modified_info_nce_loss(feats_ehr, feats_img, time_diff, k, mode)
            self.log(mode+'_loss', loss, on_step=False, on_epoch=True) #, logger=True)
            return {'loss': loss, 'feats_ehr': feats_ehr.detach().cpu(), 'feats_img': feats_img.detach().cpu(), 'y_ehr':y_ehr}
            
        else:
            if self.args.finetune:
                ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch

                ehr = torch.from_numpy(ehr).float()
                ehr = ehr.to(self.device)
                imgs = imgs.to(self.device)
                output = self.model(x=ehr, seq_lengths=seq_lengths, img=imgs)
                y_ehr = torch.from_numpy(y_ehr)
        
            else: # Features are already processed for linear classifier
                seq_lengths=None
                if 'ehr' in self.args.fusion_type:
                    ehr, y_ehr = batch
                    ehr = ehr.to(self.device)
                    output = self.model(x=ehr,seq_lengths=seq_lengths)
                elif 'cxr' in self.args.fusion_type:
                    imgs, y_cxr, y_ehr = batch
                    imgs = imgs.to(self.device)
                    output = self.model(img=imgs)
                else:
                    ehr, imgs, y_ehr, y_cxr = batch
                    ehr = ehr.to(self.device)
                    imgs = imgs.to(self.device)
                    output = self.model(x=ehr, seq_lengths=seq_lengths, img=imgs)


            if self.args.labels_set=='radiology':
                y = y_cxr.float()
                y = y.to(self.device)
            else:
                y = y_ehr.float()
                y = y.to(self.device)

            preds = output[self.args.fusion_type].squeeze()
            # Compute and log BCE loss
            if self.args.task == "length-of-stay":
                # Assuming the loss function is CrossEntropyLoss 
                y_true_bins = torch.tensor([get_bin_custom(y_item, CustomBins.nbins) for y_item in y.cpu().numpy()], dtype=torch.long).to(self.device)
                loss = self.bce_loss(preds, y_true_bins)
            else:
                loss = self.bce_loss(preds, y, mode)
            self.log(mode+'_loss', loss, on_step=False, on_epoch=True) 
            
            return {'loss': loss, 'preds': preds, 'y': y}
        
    def test_step(self, batch, batch_idx):
        mode='test'
        
        
        # Forward pass for SimCLR
        if ((self.args.fusion_type=='None') & (self.args.beta_infonce == False)):
            if self.args.beta_infonce == True:
                ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs, time_diff = batch
            else:
                ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch
            ehr = torch.from_numpy(ehr).float()
            ehr = ehr.to(self.device)
            
            # At test time of SIMCLR, always return all the layer features
            if self.args.mode == 'eval':
                feats_ehr_0, feats_ehr_3, feats_img_0, feats_img_3 = self.model(ehr, seq_lengths, imgs) 
            
                # Compute and log infoNCE loss
                if self.args.beta_infonce == True:
                    k = self.args.k
                    loss = self.modified_info_nce_loss(feats_ehr_3, feats_img_3, time_diff, k,  mode)
                else:
                    loss = self.info_nce_loss(feats_ehr_3, feats_img_3, mode)
                self.log(mode+'_loss_epoch', loss, on_step=False, on_epoch=True) #, logger=True)
            
                return {'loss': loss,   'feats_ehr_0': feats_ehr_0.detach().cpu(), 
                                        'feats_ehr_3': feats_ehr_3.detach().cpu(), 
                                        'feats_img_0': feats_img_0.detach().cpu(), 
                                        'feats_img_3': feats_img_3.detach().cpu(), 
                                        'y_ehr':y_ehr}        
        
        else:
            if self.args.finetune:
                ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch

                ehr = torch.from_numpy(ehr).float()
                ehr = ehr.to(self.device)
                imgs = imgs.to(self.device)
                output = self.model(x=ehr, seq_lengths=seq_lengths, img=imgs)
                y_ehr = torch.from_numpy(y_ehr)
            
            else: # Features are already processed for linear classifier
                seq_lengths=None
                if 'ehr' in self.args.fusion_type:
                    ehr, y_ehr = batch
                    ehr = ehr.to(self.device)
                    output = self.model(x=ehr,seq_lengths=seq_lengths)
                elif 'cxr' in self.args.fusion_type:
                    imgs, y_cxr, y_ehr = batch
                    imgs = imgs.to(self.device)
                    output = self.model(img=imgs)
                else:
                    ehr, imgs, y_ehr, y_cxr = batch
                    ehr = ehr.to(self.device)
                    imgs = imgs.to(self.device)
                    output = self.model(x=ehr, seq_lengths=seq_lengths, img=imgs)


            if self.args.labels_set=='radiology':
                y = y_cxr.float()
                y = y.to(self.device)
            else:
                y = y_ehr.float()
                y = y.to(self.device)

            preds = output[self.args.fusion_type].squeeze()
            
            # Compute and log BCE loss
            if self.args.task == "length-of-stay":
                # Assuming the loss function is CrossEntropyLoss 
                y_true_bins = torch.tensor([get_bin_custom(y_item, CustomBins.nbins) for y_item in y.cpu().numpy()], dtype=torch.long).to(self.device)
                loss = self.bce_loss(preds, y_true_bins)
            else:
                loss = self.bce_loss(preds, y, mode)
            self.log(mode+'_loss', loss, on_step=False, on_epoch=True) 
            return {'loss': loss, 'preds': preds, 'y': y}
                
    
    def process_features(self, outputs, mode):
        y = []
        if self.args.mode=='eval':
            feats_ehr_0=[]
            feats_ehr_3=[]
            feats_img_0=[]
            feats_img_3=[]
        elif self.args.fusion_type!='None':
            preds = []
        else:
            feats_ehr = []
            feats_img = []
        # Iterate through batches and append
        i=0
        for output in outputs:
            if i ==0:
                if self.args.fusion_type!='None':
                    preds = output['preds'].detach().cpu()
                elif self.args.mode == 'eval':
                    feats_ehr_0 = output['feats_ehr_0'].detach().cpu()
                    feats_ehr_3 = output['feats_ehr_3'].detach().cpu()
                    feats_img_0 = output['feats_img_0'].detach().cpu()
                    feats_img_3 = output['feats_img_3'].detach().cpu()
                else: 
                    feats_ehr = output['feats_ehr'].detach().cpu()
                    feats_img = output['feats_img'].detach().cpu()
                y = output['y'].tolist()
                
            else:
                if self.args.fusion_type!='None':
                    preds = torch.cat((preds, output['preds'].detach().cpu()))
                elif self.args.mode == 'eval':
                    feats_ehr_0 = torch.cat((feats_ehr_0, output['feats_ehr_0'].detach().cpu()))
                    feats_ehr_3 = torch.cat((feats_ehr_3, output['feats_ehr_3'].detach().cpu()))
                    feats_img_0 = torch.cat((feats_img_0, output['feats_img_0'].detach().cpu()))
                    feats_img_3 = torch.cat((feats_img_3, output['feats_img_3'].detach().cpu()))
                else:
                    feats_ehr = torch.cat((feats_ehr, output['feats_ehr'].detach().cpu()))
                    feats_img = torch.cat((feats_img, output['feats_img'].detach().cpu()))
                y.extend(output['y'].tolist())
            i+=1
        if self.args.fusion_type!='None':
            return y, preds
        elif self.args.mode=='eval':
            return feats_ehr_0, feats_ehr_3, feats_img_0, feats_img_3, y
        else:
            return feats_ehr, feats_img, y
    
    def save_features(self, x, descrip, mode):
        model_path = self.args.save_dir+'/simclr_lr/'+self.args.file_name
        if not os.path.exists(model_path):
          os.makedirs(model_path)
        
        torch.save(x, model_path+'/{}_{}_epoch_{}.pt'.format(mode, descrip, self.current_epoch))
    
    def training_epoch_end(self, outputs):
        mode='train'
        if ((self.args.fusion_type=='None') & (self.args.save_features == True)):
            feats_ehr, feats_img, y = self.process_features(outputs, mode)
            self.save_features(feats_ehr, 'feats_ehr', mode)
            self.save_features(feats_img, 'feats_img', mode)      
            self.save_features(y, 'y', mode)  
        else:
            y, preds = self.process_features(outputs, mode)

            if self.task=='length-of-stay':
                with torch.no_grad():
                    y = torch.tensor(y)
                    y_true_bins = [get_bin_custom(y_item, CustomBins.nbins) for y_item in y.cpu().numpy()]
                    pred_labels = torch.max(preds, 1)[1].cpu().numpy()  # Convert logits to predicted labels
                    cf = metrics.confusion_matrix(y_true_bins, pred_labels)
                    kappa = metrics.cohen_kappa_score(y_true_bins, pred_labels, weights='linear')
                    mad = metrics.mean_absolute_error(y.cpu().numpy(), preds.max(1)[0].cpu().numpy())
                    mse = metrics.mean_squared_error(y.cpu().numpy(), preds.max(1)[0].cpu().numpy())
                    mape = mean_absolute_percentage_error(y.cpu().numpy(), preds.max(1)[0].cpu().numpy())

                    self.log(mode + '_kappa_epoch', kappa, on_step=False, on_epoch=True)
                    self.log(mode + '_mad_epoch', mad,  on_step=False, on_epoch=True)
                    self.log(mode + '_mse_epoch', mse,  on_step=False, on_epoch=True)
                    self.log(mode + '_mape_epoch', mape,  on_step=False, on_epoch=True) 

            else:
                auroc = np.round(roc_auc_score(y, preds), 4)
                auprc = np.round(average_precision_score(y, preds), 4)
                self.log(mode + '_auroc_epoch', auroc)
                self.log(mode + '_auprc_epoch', auprc)

    
        
    def validation_epoch_end(self, outputs):
        mode='val'
        if ((self.args.fusion_type=='None') & (self.args.save_features == True)):
            feats_ehr, feats_img, y = self.process_features(outputs, mode)
            self.save_features(feats_ehr, 'feats_ehr', mode)
            self.save_features(feats_img, 'feats_img', mode)      
            self.save_features(y, 'y', mode)
        else:
            y, preds = self.process_features(outputs, mode)

            if self.task=='length-of-stay':
                with torch.no_grad():
                    y = torch.tensor(y)
                    y_true_bins = [get_bin_custom(y_item, CustomBins.nbins) for y_item in y.cpu().numpy()]
                    pred_labels = torch.max(preds, 1)[1].cpu().numpy()  # Convert logits to predicted labels
                    cf = metrics.confusion_matrix(y_true_bins, pred_labels)
                    kappa = metrics.cohen_kappa_score(y_true_bins, pred_labels, weights='linear')
                    mad = metrics.mean_absolute_error(y.cpu().numpy(), preds.max(1)[0].cpu().numpy())
                    mse = metrics.mean_squared_error(y.cpu().numpy(), preds.max(1)[0].cpu().numpy())
                    mape = mean_absolute_percentage_error(y.cpu().numpy(), preds.max(1)[0].cpu().numpy())

                    self.log(mode + '_kappa_epoch', kappa, on_step=False, on_epoch=True)
                    self.log(mode + '_mad_epoch', mad,  on_step=False, on_epoch=True)
                    self.log(mode + '_mse_epoch', mse,  on_step=False, on_epoch=True)
                    self.log(mode + '_mape_epoch', mape,  on_step=False, on_epoch=True) 

            else:
                auroc = np.round(roc_auc_score(y, preds), 4)
                auprc = np.round(average_precision_score(y, preds), 4)
                self.log(mode + '_auroc_epoch', auroc)
                self.log(mode + '_auprc_epoch', auprc)
            

    def test_epoch_end(self, outputs):
        if ((self.args.fusion_type=='None') & (self.args.save_features == True)):
            mode = self.args.eval_set
            feats_ehr_0, feats_ehr_3, feats_img_0, feats_img_3, y = self.process_features(outputs, mode)
            self.save_features(feats_ehr_0, 'feats_ehr_0', mode)
            self.save_features(feats_ehr_3, 'feats_ehr_3', mode)
            self.save_features(feats_img_0, 'feats_img_0', mode)
            self.save_features(feats_img_3, 'feats_img_3', mode)      
            self.save_features(y, 'y', mode)
        else:
            mode = 'test'
            y, preds = self.process_features(outputs, mode)

            if self.task=='length-of-stay':
                with torch.no_grad():
                    y = torch.tensor(y)
                    y_true_bins = [get_bin_custom(y_item, CustomBins.nbins) for y_item in y.cpu().numpy()]
                    pred_labels = torch.max(preds, 1)[1].cpu().numpy()  # Convert logits to predicted labels
                    cf = metrics.confusion_matrix(y_true_bins, pred_labels)
                    kappa = metrics.cohen_kappa_score(y_true_bins, pred_labels, weights='linear')
                    mad = metrics.mean_absolute_error(y.cpu().numpy(), preds.max(1)[0].cpu().numpy())
                    mse = metrics.mean_squared_error(y.cpu().numpy(), preds.max(1)[0].cpu().numpy())
                    mape = mean_absolute_percentage_error(y.cpu().numpy(), preds.max(1)[0].cpu().numpy())

                    self.log(mode + '_kappa_epoch', kappa, on_step=False, on_epoch=True)
                    self.log(mode + '_mad_epoch', mad,  on_step=False, on_epoch=True)
                    self.log(mode + '_mse_epoch', mse,  on_step=False, on_epoch=True)
                    self.log(mode + '_mape_epoch', mape,  on_step=False, on_epoch=True) 

            else:
                auroc = np.round(roc_auc_score(y, preds), 4)
                auprc = np.round(average_precision_score(y, preds), 4)
                self.log(mode + '_auroc_epoch', auroc)
                self.log(mode + '_auprc_epoch', auprc)

            if self.task =='phenotyping':    
                auroc_per_label = np.round(roc_auc_score(y, preds, average=None), 4)
                auprc_per_label = np.round(average_precision_score(y, preds, average=None), 4)
                auroc_label={}
                auprc_label={}
                for i, name in enumerate(self.LABEL_COLUMNS):
                    auroc_label[name]=auroc_per_label[i].item()
                    auprc_label[name]=auprc_per_label[i].item()
                self.log('auroc_label', auroc_label)
                self.log('auprc_label', auprc_label)
     
            
    def calculate_auroc_epoch(self, outputs, mode):
        labels = []
        predictions = []
        auroc_label={}
        outputs=outputs[self.args.fusion_type].squeeze()
        
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        for i, name in enumerate(self.LABEL_COLUMNS):
            class_roc_auc = roc_auc_score(labels[:, i], predictions[:, i])
            auroc_label[name]=class_roc_auc
            
        auroc = roc_auc_score(labels, predictions)
        auprc = average_precision_score(labels, predictions)
        
        return auroc, auprc
       
    
def return_model_version(trainer):
    filename = trainer.checkpoint_callback.filename
    best_model_path = trainer.checkpoint_callback.best_model_path
    return  filename+best_model_path.split('.ckpt')[0].split(filename)[1]

def count_parameters(model:nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, args, train_loader, val_loader, **kwargs): 
    filename = args.file_name+'_epoch_{epoch:02d}'
    
    model_path = args.save_dir+'/'+args.file_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # For model selection
    if 'logger' not in kwargs:
        print("Model selection")
        checkpoint_callback = ModelCheckpoint(monitor="val_auroc", mode='max',
                                              filename=args.load_state+'_{epoch:02d}',
                                              dirpath=args.save_dir,
                                              save_top_k=1,
                                              every_n_epochs=1,
                                              save_on_train_epoch_end=True
                                              )
        
        trainer = pl.Trainer(default_root_dir=os.path.join(model_path),
                         accelerator="auto",
                         max_epochs=args.epochs, gpus=gpu,
                         callbacks=[checkpoint_callback],
                         enable_progress_bar=False,
                         num_sanity_val_steps=0)
                         
        
    # For SIMCLR training
    else:
        print("SIMCLR training")
        logger = kwargs['logger']
        checkpoints = ModelCheckpoint(#monitor="val_auroc_epoch", mode='max',
                                    dirpath=model_path,
                                  filename=filename,
                                  save_weights_only=True, 
                                  save_top_k=1,
                                  auto_insert_metric_name=False, 
                                  every_n_epochs=1,             
                                  save_on_train_epoch_end=True)
        if args.num_gpu == 1:
            strategy = None
        else:
            strategy = 'ddp'
        if args.task== 'length-of-stay':    
            early_stop_callback = EarlyStopping(monitor="val_kappa_epoch", min_delta=0.001, mode="max", patience=30)
        elif args.fusion_type=='None':
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, mode="min", patience=30)
        else:
            early_stop_callback = EarlyStopping(monitor="val_auroc_epoch", min_delta=0.001, mode="max", patience=30)
        trainer = pl.Trainer(default_root_dir=os.path.join(model_path),
                             max_epochs=args.epochs, #gpus=gpu,
                             callbacks=[checkpoints, LearningRateMonitor('epoch'), early_stop_callback],
                             logger=logger,  
                             log_every_n_steps=5,  enable_progress_bar=True,
                             num_sanity_val_steps=0,
                            accelerator='gpu', devices=args.num_gpu, strategy=strategy)

   
    num_batches_train = len(train_loader)   
    model_parameters = count_parameters(model)
    print("Model parameters : ", model_parameters)
    trainer.fit(model, train_loader, val_loader)
        
    return trainer
    

# Call this function if doing test without training
def test(model, args, test_loader, **kwargs):
    if 'logger' not in kwargs:
        trainer = pl.Trainer(default_root_dir=os.path.join(args.save_dir))
        
    else:
        logger = kwargs['logger']
        trainer = pl.Trainer(default_root_dir=os.path.join(args.save_dir), logger=logger)
    
    trainer.test(model, test_loader)
    
    return trainer

    
# Prepare data features for downstream tasks 
@torch.no_grad()
def prepare_data_features(device, model, data_loader, bs, fusion_layer, fusion_type):
    # Prepare model
    network = deepcopy(model)
    if 'ehr' not in fusion_type:
        network.model.cxr_model.vision_backbone.fc = nn.Identity() # Removing projection head g(.) 
     
    if 'cxr' not in fusion_type:
        network.model.ehr_model.dense_layer = nn.Identity() # Removing projection head g(.)
    
    network.eval()
    network.to(device)

    # Encode all images
    feats_ehr, feats_imgs, labels_ehr, labels_imgs = [], [], [], []
    
    for batch_ehr, batch_imgs, batch_ehr_labels, batch_cxr_labels, seq_lengths, pairs in data_loader:
        labels_ehr.append(torch.from_numpy(batch_ehr_labels).detach())
        
        if 'cxr' not in fusion_type:
            batch_ehr = torch.from_numpy(batch_ehr).float().to(device)
            batch_ehr_feats = network.model.ehr_model(batch_ehr, seq_lengths)
            if fusion_layer == 3:
                batch_ehr_feats = network.model.ehr_model_g(batch_ehr_feats)
            feats_ehr.append(batch_ehr_feats.detach().cpu()) 

        if 'ehr' not in fusion_type:
            batch_imgs = batch_imgs.to(device)
            batch_imgs_feats = network.model.cxr_model(batch_imgs)
            if fusion_layer == 3:
                batch_imgs_feats = network.model.cxr_model_g(batch_imgs_feats)
                
            feats_imgs.append(batch_imgs_feats.detach().cpu())
            labels_imgs.append(batch_cxr_labels)
    
    labels_ehr = torch.cat(labels_ehr, dim=0)
    
    if 'cxr' not in fusion_type:
        feats_ehr = torch.cat(feats_ehr, dim=0)
        

    if 'ehr' not in fusion_type:
        feats_imgs = torch.cat(feats_imgs, dim=0)
        labels_imgs = torch.cat(labels_imgs, dim=0)

    if 'cxr' in fusion_type:
        return data.DataLoader(data.TensorDataset(feats_imgs, labels_imgs, labels_ehr), batch_size=bs, shuffle=False, drop_last=False, pin_memory=True, num_workers=16)
    elif 'ehr' in fusion_type:
        return data.DataLoader(data.TensorDataset(feats_ehr, labels_ehr), batch_size=bs, shuffle=False, drop_last=False, pin_memory=True, num_workers=16)
    else:
        return data.DataLoader(data.TensorDataset(feats_ehr, feats_imgs, labels_ehr, labels_imgs), batch_size=bs, shuffle=False, pin_memory=True, drop_last=False, num_workers=16)

            
