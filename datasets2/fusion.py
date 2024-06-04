import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
# import 
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
from datetime import timedelta

R_CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']

CLASSES = [
       'Acute and unspecified renal failure', 'Acute cerebrovascular disease',
       'Acute myocardial infarction', 'Cardiac dysrhythmias',
       'Chronic kidney disease',
       'Chronic obstructive pulmonary disease and bronchiectasis',
       'Complications of surgical procedures or medical care',
       'Conduction disorders', 'Congestive heart failure; nonhypertensive',
       'Coronary atherosclerosis and other heart disease',
       'Diabetes mellitus with complications',
       'Diabetes mellitus without complication',
       'Disorders of lipid metabolism', 'Essential hypertension',
       'Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage',
       'Hypertension with complications and secondary hypertension',
       'Other liver diseases', 'Other lower respiratory disease',
       'Other upper respiratory disease',
       'Pleurisy; pneumothorax; pulmonary collapse',
       'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
       'Respiratory failure; insufficiency; arrest (adult)',
       'Septicemia (except in labor)', 'Shock'
    ]

class MIMIC_CXR_EHR(Dataset):
    def __init__(self, args, metadata_with_labels, ehr_ds, cxr_ds, split='train'):
        
        # select classes
        self.CLASSES = CLASSES
        if 'radiology' in args.labels_set:
            self.CLASSES = R_CLASSES
        
        self.metadata_with_labels = metadata_with_labels
        
        self.cxr_files_paired = self.metadata_with_labels.dicom_id.values
        self.ehr_files_paired = (self.metadata_with_labels['stay'].values)
        self.time_diff = self.metadata_with_labels.time_diff
        self.lower = self.metadata_with_labels.lower
        self.upper = self.metadata_with_labels.upper
        
        self.cxr_files_all = cxr_ds.filenames_loaded
        self.ehr_files_all = ehr_ds.names
        
        self.ehr_files_unpaired = list(set(self.ehr_files_all) - set(self.ehr_files_paired))
        
        self.ehr_ds = ehr_ds
        self.cxr_ds = cxr_ds
        
        self.args = args
        self.split = split
        self.data_ratio = self.args.data_ratio if split=='train' else 1.0

    def __getitem__(self, index):
        if self.args.data_pairs == 'paired':
            cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_paired[index]]
            
            lower = self.metadata_with_labels.iloc[index].lower
            upper = self.metadata_with_labels.iloc[index].upper
            
            ehr_data, labels_ehr = self.ehr_ds.__getitem__(self.ehr_files_paired[index],lower,upper)
            time_diff = self.metadata_with_labels.iloc[index].time_diff
            
            #dicom_id =  self.cxr_files_paired[index]
            #stay_id = self.ehr_files_paired[index]
            #time_diff = 
                        
            if self.args.beta_infonce:
                return ehr_data, cxr_data, labels_ehr, labels_cxr, time_diff
            else:
                return ehr_data, cxr_data, labels_ehr, labels_cxr
        
        elif self.args.data_pairs == 'radiology':
            ehr_data, labels_ehr = np.zeros((1, 10)), np.zeros(self.args.num_classes)
            cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_all[index]]
            return ehr_data, cxr_data, labels_ehr, labels_cxr
        
        elif self.args.data_pairs == 'ehr_only':
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_all[index]]
            cxr_data, labels_cxr = None, None
            return ehr_data, cxr_data, labels_ehr, labels_cxr
        
        elif self.args.data_pairs == 'joint_ehr':
            if index < len(self.ehr_files_paired):
                ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
                cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_paired[index]]
            else:
                index = random.randint(0, len(self.ehr_files_unpaired)-1) 
                ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_unpaired[index]]
                cxr_data, labels_cxr = None, None
            return ehr_data, cxr_data, labels_ehr, labels_cxr

       
    
    def __len__(self):
        if self.args.data_pairs == 'paired':
            return len(self.ehr_files_paired)
        elif self.args.data_pairs == 'ehr_only':
            return len(self.ehr_files_all)
        elif self.args.data_pairs == 'radiology':
            return len(self.cxr_files_all)
        elif self.args.data_pairs == 'joint_ehr':
            return len(self.ehr_files_paired) + int(self.data_ratio * len(self.ehr_files_unpaired)) 
        


def loadmetadata(args):

    def time_offsets(table):
        ids = table.stay_id.unique()
        data = []
        for id in ids:
            temp = table[table.stay_id == id]
            offsets = list(range(0,int(temp.LOS.max())+12,12))
            times = []
            for i,time in enumerate(temp.intime):
                times.append(time+ timedelta(hours=offsets[i])) 
            temp.intime = times
            data.append(temp)
        data = pd.concat(data,ignore_index=True)
        data.time_diff = np.abs((data.StudyDateTime - data.intime).apply(lambda x: np.round(x.total_seconds()/60/60,3)))
        data['lower'] = data.LOS + (data.intime - data.outtime).apply(lambda x: np.round(x.total_seconds()/60/60,3))
        data['upper'] = data.apply(lambda x: x.lower + 12 if (x.lower + 12) < x.LOS else (x.LOS+1),axis=1)
        return data
    
    cxr_metadata = pd.read_csv(f'{args.cxr_data_root}/mimic-cxr-2.0.0-metadata.csv')
    print('Number of CXR images=', len(cxr_metadata))
    icu_stay_metadata = pd.read_csv(f'{args.ehr_data_root}/root/all_stays.csv')
    print('Number of ICU stays=', len(icu_stay_metadata))
    columns = ['subject_id', 'stay_id', 'intime', 'outtime']
    
    # only common subjects with both icu stay and an xray
    # Note that inner merge includes rows if a chest X-ray is associated with multiple stays
    cxr_merged_icustays = cxr_metadata.merge(icu_stay_metadata[columns], how='inner', on='subject_id')
    print('Number of CXR associated with ICU stay based on subject ID=', len(cxr_merged_icustays))
    print('Number of unique CXR dicoms=', len(cxr_merged_icustays.dicom_id.unique()))
    print('Number of unique CXR study id=', len(cxr_merged_icustays.study_id.unique()))
        
    # combine study date time
    cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
    cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(cxr_merged_icustays['StudyDate'].astype(str) + ' ' + cxr_merged_icustays['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")

    cxr_merged_icustays.intime=pd.to_datetime(cxr_merged_icustays.intime)
    cxr_merged_icustays.outtime=pd.to_datetime(cxr_merged_icustays.outtime)
    
    cxr_merged_icustays['time_diff'] = cxr_merged_icustays.StudyDateTime-cxr_merged_icustays.intime
    cxr_merged_icustays['time_diff'] = cxr_merged_icustays['time_diff'].apply(lambda x: np.round(x.total_seconds()/60/60,3))
    
    cxr_merged_icustays['LOS'] = cxr_merged_icustays.outtime-cxr_merged_icustays.intime
    cxr_merged_icustays['LOS'] = cxr_merged_icustays['LOS'].apply(lambda x: np.round(x.total_seconds()/60/60,3))
    
    
    # For LE/ FT  (evaluation datasets)
    if (args.dataset !='all'):

        if args.task == 'decompensation' or args.task == 'length-of-stay':
            train_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/train_listfile.csv')
            train_listfile.columns = ['stay' , 'period_length' , 'stay_id' ,'y_true', 'intime' , 'endtime']
            test_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/test_listfile.csv')
            test_listfile.columns = ['stay' , 'period_length' , 'stay_id' ,'y_true', 'intime' , 'endtime']
            listfile = train_listfile.append(test_listfile)
            listfile['subject_id'] = listfile['stay'].apply(lambda x: x.split("_")[0])
            print(listfile.head)

            columns2 = ['subject_id', 'endtime']
            listfile['subject_id'] = listfile['subject_id'].astype('int64')
            cxr_merged_icustays = cxr_merged_icustays.merge(listfile[columns2], how='inner', on='subject_id')
            cxr_merged_icustays.endtime=pd.to_datetime(cxr_merged_icustays.endtime)
            cxr_merged_icustays_during = cxr_merged_icustays.loc[((cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&(cxr_merged_icustays.StudyDateTime<=cxr_merged_icustays.endtime))]

        if args.task == 'in-hospital-mortality':
            end_time = cxr_merged_icustays.intime + pd.DateOffset(hours=48)
            cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&((cxr_merged_icustays.StudyDateTime<=end_time))]

        if args.task == 'phenotyping':
            end_time = cxr_merged_icustays.outtime
            cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&
                                                                 ((cxr_merged_icustays.StudyDateTime<=end_time))]
        
        # select cxrs with the ViewPosition == 'AP
        cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
        
        if args.retrive_cxr == 'recent':
            groups = cxr_merged_icustays_AP.groupby('stay_id')
            groups_selected = []
            for group in groups:
                # select the latest cxr for the icu stay
                selected = group[1].sort_values('StudyDateTime').tail(1).reset_index()
                groups_selected.append(selected)
            groups = pd.concat(groups_selected, ignore_index=True)
            groups = groups.groupby('study_id').first()
            groups = groups.reset_index()
            groups = groups.groupby('study_id').first().sort_values(by=['stay_id','StudyDateTime'])
            groups = groups.reset_index()
            #groups['num_cxr_windows'] = groups.groupby(['stay_id'])['stay_id'].transform('count')
            #groups['cxr_window_length'] = groups['LOS']/groups['num_cxr_windows']
            groups['num_ehr_windows'] = np.ceil(groups['LOS']/12).astype(int)
            groups = groups.loc[groups.index.repeat(groups.num_ehr_windows)].reset_index(drop=True)
            groups = time_offsets(groups)
        else: 
            groups = cxr_merged_icustays_AP.groupby('study_id').first()
            groups = groups.reset_index()
            groups = groups.groupby('study_id').first().sort_values(by=['stay_id','StudyDateTime'])
            groups = groups.reset_index()
            #groups['num_cxr_windows'] = groups.groupby(['stay_id'])['stay_id'].transform('count')
            #groups['cxr_window_length'] = groups['LOS']/groups['num_cxr_windows']
            groups['num_ehr_windows'] = np.ceil(groups['LOS']/12).astype(int)
            
            
    # For SIMCLR pretraining (large dataset)
#     else:
#         # print(cxr_merged_icustays.ViewPosition.unique())
#         cxr_merged_icustays_AP = cxr_merged_icustays[cxr_merged_icustays['ViewPosition'] == 'AP']
#         print("Number of CXR associated with ICU stay and in AP view=", len(cxr_merged_icustays_AP))
#         groups = cxr_merged_icustays_AP
        
    print("Mean time cxr - intime= ", groups.time_diff.mean())
    print("Minimum time =", groups.time_diff.min())
    print("Maximum time =", groups.time_diff.max())

#     plt.hist(groups.time_diff.apply(lambda x: x.days).astype("float64"))
#     plt.xlabel('Time difference in days')
#     plt.show()

    #print(groups.iloc[0])
    return groups
    
def load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds):
    
    # Load cxr and ehr groups
    cxr_merged_icustays = loadmetadata(args) 
    
    # Add the labels 
    splits_labels_train = pd.read_csv(f'{args.ehr_data_root}/{args.task}/train_listfile.csv')
    splits_labels_val = pd.read_csv(f'{args.ehr_data_root}/{args.task}/val_listfile.csv')
    splits_labels_test = pd.read_csv(f'{args.ehr_data_root}/{args.task}/test_listfile.csv')
    
    #TODO: investigate why total size of cxr_merged_icustays drops after the three steps below
    train_meta_with_labels = cxr_merged_icustays.merge(splits_labels_train, how='inner', on='stay_id')#change dataset size here
    val_meta_with_labels = cxr_merged_icustays.merge(splits_labels_val, how='inner', on='stay_id')
    test_meta_with_labels = cxr_merged_icustays.merge(splits_labels_test, how='inner', on='stay_id')
    
    # Get rid of chest X-rays that don't have radiology reports
    metadata = pd.read_csv(f'{args.cxr_data_root}/mimic-cxr-2.0.0-metadata.csv')
    labels = pd.read_csv(f'{args.cxr_data_root}/mimic-cxr-2.0.0-chexpert.csv')
    metadata_with_labels = metadata.merge(labels[['study_id']], how='inner', on='study_id').drop_duplicates(subset=['dicom_id'])
    train_meta_with_labels = train_meta_with_labels.merge(metadata_with_labels[['dicom_id']], how='inner', on='dicom_id')
    val_meta_with_labels = val_meta_with_labels.merge(metadata_with_labels[['dicom_id']], how='inner', on='dicom_id')
    test_meta_with_labels = test_meta_with_labels.merge(metadata_with_labels[['dicom_id']], how='inner', on='dicom_id')
    
    print("Excluding CXR with missing radiology reports = ",len(train_meta_with_labels))

    # Multimodal class
    train_ds = MIMIC_CXR_EHR(args, train_meta_with_labels, ehr_train_ds, cxr_train_ds)
    print(len(train_ds))
    val_ds = MIMIC_CXR_EHR(args, val_meta_with_labels, ehr_val_ds, cxr_val_ds, split='val')
    print(len(val_ds))
    test_ds = MIMIC_CXR_EHR(args, test_meta_with_labels, ehr_test_ds, cxr_test_ds, split='test')
    print(len(test_ds))
    
    if args.beta_infonce:
        collate = my_collate_beta
    else:
        collate = my_collate
    
    # Multimodal data loader 
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)#, pin_memory=True, num_workers=24)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate, drop_last=False) #pin_memory=True, num_workers=16,
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=collate, drop_last=False) # pin_memory=True,num_workers=16,

    return train_dl, val_dl, test_dl



def my_collate(batch):
    x = [item[0] for item in batch]
    pairs = [False if item[1] is None else True for item in batch]
    img = torch.stack([torch.zeros(3, 224, 224) if item[1] is None else item[1] for item in batch])
    x, seq_length = pad_zeros(x)
    targets_ehr = np.array([item[2] for item in batch])
    targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
    return [x, img, targets_ehr, targets_cxr, seq_length, pairs]
    
def my_collate_beta(batch, beta_infonce=False):
    x = [item[0] for item in batch]
    pairs = [False if item[1] is None else True for item in batch]
    img = torch.stack([torch.zeros(3, 224, 224) if item[1] is None else item[1] for item in batch])
    x, seq_length = pad_zeros(x)
    targets_ehr = np.array([item[2] for item in batch])
    targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
    time_diff = [item[4] for item in batch]
    
    return [x, img, targets_ehr, targets_cxr, seq_length, pairs, time_diff]
    


def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length
