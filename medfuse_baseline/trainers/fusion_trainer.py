from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.fusion import Fusion
from models.ehr_models import LSTM
from models.cxr_models import CXRModels
from .trainer import Trainer
import pandas as pd


import numpy as np
from sklearn import metrics
import wandb

class CustomBins:
    inf = 1e18
    bins = [(-1*inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, inf)]
    nbins = len(bins)
    means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
             131.579534, 155.643957, 179.660558, 254.306624, 585.325890]
             
    
def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0]*24.0
        b = CustomBins.bins[i][1]*24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None
    
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100

class FusionTrainer(Trainer):
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl
        ):

        super(FusionTrainer, self).__init__(args)
        run = wandb.init(project=f'Medfuse_{self.args.fusion_type}', config=args)
        self.epoch = 0 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        

        self.ehr_model = LSTM(args, input_dim=76, num_classes=args.num_classes, hidden_dim=args.dim, dropout=args.dropout, layers=args.layers).to(self.device)
        self.cxr_model = CXRModels(self.args, self.device).to(self.device)


        self.model = Fusion(args, self.ehr_model, self.cxr_model ).to(self.device)
        self.init_fusion_method()

        if self.args.task=="length-of-stay":
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.BCELoss()

        self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        self.load_state()
        print(self.ehr_model)
        print(self.optimizer)
        print(self.loss)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_kappa = 0
        self.best_stats = None
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99) 
        self.epochs_stats = {'loss train': [], 'loss val': [], 'auroc val': [], 'loss align train': [], 'loss align val': []}
    
    def init_fusion_method(self):

        '''
        for early fusion
        load pretrained encoders and 
        freeze both encoders
        ''' 

        if self.args.load_state_ehr is not None:
            self.load_ehr_pheno(load_state=self.args.load_state_ehr)
        if self.args.load_state_cxr is not None:
            self.load_cxr_pheno(load_state=self.args.load_state_cxr)
        
        if self.args.load_state is not None:
            print("jerry")
            self.load_state()


        if 'uni_ehr' in self.args.fusion_type:
            self.freeze(self.model.cxr_model)
        elif 'uni_cxr' in self.args.fusion_type:
            self.freeze(self.model.ehr_model)
        elif 'late' in self.args.fusion_type:
            self.freeze(self.model)
        elif 'early' in self.args.fusion_type:
            self.freeze(self.model.cxr_model)
            self.freeze(self.model.ehr_model)
        elif 'lstm' in self.args.fusion_type:
            # self.freeze(self.model.cxr_model)
            # self.freeze(self.model.ehr_model)
            pass

    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)

            output = self.model(x, seq_lengths, img, pairs)
            
            pred = output[self.args.fusion_type].squeeze()
            
            if self.args.task == "length-of-stay":
                # Assuming the loss function is CrossEntropyLoss which needs class indices
                y_true_bins = torch.tensor([get_bin_custom(y_item, CustomBins.nbins) for y_item in y.cpu().numpy()], dtype=torch.long).to(self.device)
                loss = self.loss(pred, y_true_bins)
            else:
                loss = self.loss(pred, y)
            
            epoch_loss += loss.item()
            if self.args.align > 0.0:
                loss = loss + self.args.align * output['align_loss']
                epoch_loss_align = epoch_loss_align + self.args.align * output['align_loss'].item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f} loss align {epoch_loss_align/i:0.4f}")
        
        if self.args.task == "length-of-stay":
            with torch.no_grad():
                y_true_bins = [get_bin_custom(y_item.item(), CustomBins.nbins) for y_item in outGT.cpu().numpy()]
                pred_labels = torch.max(outPRED, 1)[1].cpu().numpy()  # Convert logits to predicted labels
                cf = metrics.confusion_matrix(y_true_bins, pred_labels)
                kappa = metrics.cohen_kappa_score(y_true_bins, pred_labels, weights='linear')
                mad = metrics.mean_absolute_error(outGT.cpu().numpy(), outPRED.max(1)[0].cpu().numpy())
                mse = metrics.mean_squared_error(outGT.cpu().numpy(), outPRED.max(1)[0].cpu().numpy())
                mape = mean_absolute_percentage_error(outGT.cpu().numpy(), outPRED.max(1)[0].cpu().numpy())
    
                best_stats = {"mad": mad, "mse": mse, "mape": mape, "kappa": kappa}
                wandb.log({
                    'train_mad': mad,
                    'train_mse': mse, 
                    'train_mape': mape,
                    'train_kappa': kappa
                })
                ret = best_stats
        else:    
            ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
            self.epochs_stats['loss train'].append(epoch_loss/i)
            self.epochs_stats['loss align train'].append(epoch_loss_align/i)
            wandb.log({
                    'train_Loss': epoch_loss/i, 
                    'train_AUC': ret['auroc_mean']
                })
        return ret
    
    def validate(self, dl):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0
        # ehr_features = torch.FloatTensor()
        # cxr_features = torch.FloatTensor()
        outGT = torch.FloatTensor().to(self.device)
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (dl):
                #print(y_ehr)
                y = self.get_gt(y_ehr, y_cxr)

                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                y = Variable(y.to(self.device), requires_grad=False)
                img = img.to(self.device)
                output = self.model(x, seq_lengths, img, pairs)
                
                pred = output[self.args.fusion_type]
                
                if self.args.fusion_type != 'uni_cxr':
                    if len(pred.shape) > 1:
                         pred = pred.squeeze()
                if self.args.task == "length-of-stay":
                    #print(y)
                    y_true_bins = torch.tensor([get_bin_custom(y_item, CustomBins.nbins) for y_item in y.cpu().numpy()], dtype=torch.long).to(self.device)
                    loss = self.loss(pred, y_true_bins)
                    #print(y_true_bins)
                    #print(pred)
                else:
                    loss = self.loss(pred, y)
                epoch_loss += loss.item()
                if self.args.align > 0.0:
                    epoch_loss_align +=  output['align_loss'].item()
                outPRED = torch.cat((outPRED, pred), 0)
                outGT = torch.cat((outGT, y), 0)
                # if 'ehr_feats' in output:
                #     ehr_features = torch.cat((ehr_features, output['ehr_feats'].data.cpu()), 0)
                # if 'cxr_feats' in output:
                #     cxr_features = torch.cat((cxr_features, output['cxr_feats'].data.cpu()), 0)
        
        self.scheduler.step(epoch_loss/len(self.val_dl))

        print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f} \t{epoch_loss_align/i:0.5f}")
        if self.args.task == "length-of-stay":
            with torch.no_grad():
                y_true_bins = [get_bin_custom(y_item.item(), CustomBins.nbins) for y_item in outGT.cpu().numpy()]
                pred_labels = torch.max(outPRED, 1)[1].cpu().numpy()  # Convert logits to predicted labels
                cf = metrics.confusion_matrix(y_true_bins, pred_labels)
                kappa = metrics.cohen_kappa_score(y_true_bins, pred_labels, weights='linear')
                mad = metrics.mean_absolute_error(outGT.cpu().numpy(), outPRED.max(1)[0].cpu().numpy())
                mse = metrics.mean_squared_error(outGT.cpu().numpy(), outPRED.max(1)[0].cpu().numpy())
                mape = mean_absolute_percentage_error(outGT.cpu().numpy(), outPRED.max(1)[0].cpu().numpy())
    
                best_stats = {"mad": mad, "mse": mse, "mape": mape, "kappa": kappa}
                wandb.log({
                    'val_mad': mad,
                    'val_mse': mse, 
                    'val_mape': mape,
                    'val_kappa': kappa
                })
            ret = best_stats
        else:    
            ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy()) 
            np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy())
            self.epochs_stats['auroc val'].append(ret['auroc_mean'])
            self.epochs_stats['loss val'].append(epoch_loss/i)
            self.epochs_stats['loss align val'].append(epoch_loss_align/i)
        # print(f'true {outGT.data.cpu().numpy().sum()}/{outGT.data.cpu().numpy().shape}')
        # print(f'true {outGT.data.cpu().numpy().sum()/outGT.data.cpu().numpy().shape[0]} ({outGT.data.cpu().numpy().sum()}/{outGT.data.cpu().numpy().shape[0]})')
            wandb.log({
                    'val_Loss': epoch_loss/i, 
                    'val_AUC': ret['auroc_mean']
                })

        return ret

    
    def compute_late_fusion(self, y_true, uniout_cxr, uniout_ehr):
        y_true = np.array(y_true)
        predictions_cxr = np.array(uniout_cxr)
        predictions_ehr = np.array(uniout_ehr)
        best_weights = np.ones(y_true.shape[-1])
        best_auroc = 0.0
        weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for class_idx in range(y_true.shape[-1]):
            for weight in weights:
                predictions = (predictions_ehr * best_weights) + (predictions_cxr * (1-best_weights))
                predictions[:, class_idx] = (predictions_ehr[:, class_idx] * weight) + (predictions_cxr[:, class_idx] * 1-weight)
                auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
                auroc_mean = np.mean(np.array(auc_scores))
                if auroc_mean > best_auroc:
                    best_auroc = auroc_mean
                    best_weights[class_idx] = weight
                # predictions = weight * predictions_cxr[]


        predictions = (predictions_ehr * best_weights) + (predictions_cxr * (1-best_weights))
        print(best_weights)

        if args.task in ['phenotyping', 'in-hospital-mortality', 'decompensation']:

            auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
            ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                                average="micro")
            ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                                average="macro")
            ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                                    average="weighted")
            
            # print(np.mean(np.array(auc_scores)

            # print
            best_stats = {"auc_scores": auc_scores,
                    "ave_auc_micro": ave_auc_micro,
                    "ave_auc_macro": ave_auc_macro,
                    "ave_auc_weighted": ave_auc_weighted,
                    "auroc_mean": np.mean(np.array(auc_scores))
                    }

        elif args.task == 'length-of-stay':
             predictions = np.maximum(predictions, 0).flatten()
             y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
             prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
             cf = metrics.confusion_matrix(y_true_bins, prediction_bins)

             kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins,
                                      weights='linear')
             mad = metrics.mean_absolute_error(y_true, predictions)
             mse = metrics.mean_squared_error(y_true, predictions)
             mape = mean_absolute_percentage_error(y_true, predictions)
             best_stats = {"mad": mad,
             "mse": mse,
             "mape": mape,
             "kappa": kappa}

        else:
            print("Task not implemented")

        #self.print_and_write(best_stats , isbest=True, prefix='late fusion weighted average')
        return best_stats 
        
    def eval_age(self):

        print('validating ... ')
           
        patiens = pd.read_csv('data/physionet.org/files/mimic-iv-1.0/core/patients.csv')
        subject_ids = np.array([int(item.split('_')[0]) for item in self.test_dl.dataset.ehr_files_paired])

        selected = patiens[patiens.subject_id.isin(subject_ids)]
        start = 18
        copy_ehr = np.copy(self.test_dl.dataset.ehr_files_paired)
        copy_cxr = np.copy(self.test_dl.dataset.cxr_files_paired)
        self.model.eval()
        step = 20
        for i in range(20, 100, step):
            subjects = selected.loc[((selected.anchor_age >= start) & (selected.anchor_age < i + step))].subject_id.values
            indexes = [jj for (jj, subject) in enumerate(subject_ids) if  subject in subjects]
            
            
            self.test_dl.dataset.ehr_files_paired = copy_ehr[indexes]
            self.test_dl.dataset.cxr_files_paired = copy_cxr[indexes]

            print(len(indexes))
            ret = self.validate(self.test_dl)
            print(f"{start}-{i + step} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")

            #self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename=f'results_test_{start}_{i + step}.txt')

            # print(f"{start}-{i + step} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")
            # print(f"{start}-{i + 10} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")
            # self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} age_{start}_{i + 10}_{len(indexes)}', filename='results_test.txt')
            start = i + step
    def test(self):
        print('validating ... ')
        self.epoch = 0
        self.model.eval()
        ret = self.validate(self.val_dl)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val.txt')
        self.model.eval()
        ret = self.validate(self.test_dl)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test.txt')
        return

    def eval(self):
        # self.eval_age()

        self.load_ehr_pheno(load_state=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.task}_{self.args.lr}_checkpoint.pth.tar')
        self.load_cxr_pheno(load_state=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.task}_{self.args.lr}_checkpoint.pth.tar')
        self.load_state(state_path=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.task}_{self.args.lr}_checkpoint.pth.tar')
        
        # self.load_ehr_pheno(load_state=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.lr}_checkpoint.pth.tar')
        # self.load_cxr_pheno(load_state=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.lr}_checkpoint.pth.tar')
        # self.load_state(state_path=f'{self.args.save_dir}/best_{self.args.fusion_type}_{self.args.lr}_checkpoint.pth.tar')
        
        self.epoch = 0
        self.model.eval()
        # ret = self.validate(self.val_dl)
        # self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val.txt')
        # self.model.eval()
        ret = self.validate(self.test_dl)
        if self.args.task=="length-of-stay":
            wandb.log({
                    'test mad': ret['mad'], 
                    'test mse': ret['mse'], 
                    'test mape': ret['mape'],
                    'test kappa': ret['kappa']
                })
        else:
            wandb.log({
                    'test_auprc': ret['auprc_mean'], 
                    'test_AUC': ret['auroc_mean']
                })
        #self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test.txt')
        return
    
    def train(self):
        print(f'running for fusion_type {self.args.fusion_type}')
        for self.epoch in range(self.start_epoch, self.args.epochs):
            self.model.eval()
            ret = self.validate(self.val_dl)
            self.save_checkpoint(prefix='last')
            
            if self.args.task=="length-of-stay":
                if self.best_kappa < ret['kappa']:
                    self.best_kappa = ret['kappa']
                    self.best_stats = ret
                    self.save_checkpoint()
                    # print(f'saving best AUROC {ret["ave_auc_micro"]:0.4f} checkpoint')
                    #self.print_and_write(ret, isbest=True)
                    self.patience = 0
                else:
                    #self.print_and_write(ret, isbest=False)
                    self.patience+=1
            else:
                if self.best_auroc < ret['auroc_mean']:
                    self.best_auroc = ret['auroc_mean']
                    self.best_stats = ret
                    self.save_checkpoint()
                    # print(f'saving best AUROC {ret["ave_auc_micro"]:0.4f} checkpoint')
                    #self.print_and_write(ret, isbest=True)
                    self.patience = 0
                else:
                    #self.print_and_write(ret, isbest=False)
                    self.patience+=1

            self.model.train()
            self.train_epoch()
            # self.plot_stats(key='loss', filename='loss.pdf')
            # self.plot_stats(key='auroc', filename='auroc.pdf')
            # if self.patience >= self.args.patience:
            #     break
        #self.print_and_write(self.best_stats , isbest=True)

        
    

