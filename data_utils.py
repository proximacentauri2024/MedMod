import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datetime import datetime
import os
from glob import glob
import pickle
from torch.utils.data import DataLoader



def load_patient_ids(path):
    return pd.read_csv(path)['stay'].astype(str).str.split('_').str[0].unique()

def process_metadata(metadata_path, stays_path, data_dir, task='decompensation'):
    metadata = load_metadata_labels(metadata_path, stays_path)
    if task == 'decompensation':
        print('Processing metadata for decompensation task')
        return metadata
    elif task == 'phenotyping':
        print('Processing metadata for phenotyping task')
        
        metadata_pheno = pd.read_csv(data_dir+'{}/train_listfile.csv'.format(task))
        metadata_pheno = pd.concat([metadata_pheno, pd.read_csv(data_dir+'{}/val_listfile.csv'.format(task))])
        metadata_pheno = pd.concat([metadata_pheno, pd.read_csv(data_dir+'{}/test_listfile.csv'.format(task))])


        metadata_pheno['subject_id'] = metadata_pheno['stay'].astype(str).str.split('_').str[0]
        metadata_pheno['stay_num'] = metadata_pheno['stay'].astype(str).str.split('episode').str[1]
        metadata_pheno['stay_num'] = metadata_pheno['stay_num'].astype(str).str.split('_').str[0].astype(int)
        
        metadata_pheno = metadata_pheno.drop(['subject_id'], axis=1)
        metadata = metadata.merge(metadata_pheno.drop_duplicates('stay_id'), how='left', on='stay_id')
        
        return metadata.loc[metadata.stay_id.notna()]


def load_metadata_labels(metadata_path, stays_path):
    
    metadata = pd.read_csv(metadata_path)
    stays = pd.read_csv(stays_path)
    
    metadata = metadata.merge(stays[['subject_id', 'deathtime']].drop_duplicates('subject_id'), how='left', on='subject_id')
    metadata['StudyTime']=metadata['StudyTime'].astype(str).str.split('.').str[0]
    metadata['StudyTime'] = metadata['StudyTime'].apply(lambda x: '00'+ x if len(x)==4 else x )
    metadata['StudyTime'] = metadata['StudyTime'].apply(lambda x: '000'+ x if len(x)==3 else x )
    metadata['StudyTime'] = metadata['StudyTime'].apply(lambda x: '0000'+ x if len(x)==2 else x )
    metadata['StudyTime'] = metadata['StudyTime'].apply(lambda x: '00000'+ x if len(x)==1 else x )

    metadata['datetime'] = metadata['StudyDate'].astype(str) + ' ' + metadata['StudyTime'].astype(str)
    metadata['datetime'] = metadata['datetime'].apply(lambda x:  datetime.strptime(x, '%Y%m%d %H%M%S') if len(x)> 13 else 0)

    metadata['time_to_death'] = pd.to_datetime(metadata['deathtime'])- metadata['datetime']

    metadata['time_to_death']= metadata.time_to_death.apply(lambda x: x.total_seconds()/60/60 if x!= np.nan else x)
    metadata['label'] = metadata['time_to_death'].apply(lambda x: 1 if ((x<=24)&(x!=np.nan)) else 0)
    
    stays.intime=pd.to_datetime(stays.intime)
    stays.outtime=pd.to_datetime(stays.outtime)
    stays.admittime=pd.to_datetime(stays.admittime)
    
    metadata_temp=metadata.merge(stays[['subject_id', 'stay_id', 'intime', 'outtime', 'admittime']], how='inner', on='subject_id')
    metadata_temp = metadata_temp.loc[(metadata_temp.datetime>=metadata_temp.intime)&((metadata_temp.datetime<=metadata_temp.outtime))]
    
    metadata = metadata.merge(metadata_temp[['study_id', 'stay_id','intime', 'outtime', 'admittime']].drop_duplicates('study_id'),
                             how='left', on='study_id')
    
    return metadata


def save_metadata_for_loader(metadata, path, file_name, task='decompensation', labels=[]):
    #print('saving 1000 rows only')
    
    if task == 'decompensation':
        df = metadata.drop_duplicates('study_id')[['subject_id', 'study_id', 'stay_id', 'datetime','time_to_death', 'label']]#.iloc[0:1000]
    
    elif task == 'phenotyping':
        
        df = metadata.drop_duplicates('study_id')[['subject_id', 'study_id', 'datetime', 'stay_id']+labels]#.iloc[0:1000]
        df = df.dropna()
    
    df.to_csv(path+file_name)
    return 


def display_image_loader(dataloader, task='decompensation'):
    # Display image and label.
    features, labels = next(iter(dataloader))
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    img = features[0].squeeze()
    label = labels[0]
    plt.imshow(img[0], cmap="gray")
    plt.show()
    if task == 'decompensation':
        print(f"Label: {label}")
    elif task == 'phenotyping':
        print(f"Labels: {label}")

        
# Normalization function from covid/us projects
class Standardizer(object):

    def __init__(self):
        pass

    def __call__(self, img):
        img = (img - img.mean()) / np.maximum(img.std(), 10 ** (-5))
        return img
    
    
    

    
## dataloader for the MIMIC dataset 
class MIMICCXR(Dataset):
    def __init__(self, metadata_file, img_dir, transform=None, target_transform=None, random=False, task='decompensation', labels=[], transform_flag =0):
        self.metadata = pd.read_csv(metadata_file)
        if task == 'decompensation':
            self.img_labels = self.metadata.label.values
        elif task == 'phenotyping':
            self.img_labels = self.metadata[labels].values
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.random = random
        self.transform_flag = transform_flag
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, 'p'+self.metadata.iloc[idx]['subject_id'].astype(str)[0:2] +'/p'+self.metadata.iloc[idx]['subject_id'].astype(str)+'/s'+self.metadata.iloc[idx]['study_id'].astype(str)+'/')
        imgs = glob(img_path+"/*jpg")
        
        self.stay_id = self.metadata.iloc[idx]['stay_id']
        self.study_id = self.metadata.iloc[idx]['study_id']
        
        if len(imgs)>0:
            if self.random == True:
                img_path = np.random.choice() # choose the image randomly 
            else:
                img_path = imgs[0]# choose the first image 
        else:
            print(img_path)
     
        # From CheXclusion
        image = Image.open(img_path)
        image = transforms.ToTensor()(image)
        
        if self.transform_flag == 0:
            image = transforms.Resize(256)(image)
            image = transforms.CenterCrop(224)(image) 
        elif self.transform_flag == 1:
            image = transforms.Resize(256)(image)
            image = transforms.RandomHorizontalFlip(p=0.5)(image)
            #image = transforms.RandomVerticalFlip(p=0.5)(image)
            image = transforms.RandomAffine(degrees=(-10, 10), translate=(0.1,0.1), fillcolor=0)(image) 
            #RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25))(image)
            image = transforms.CenterCrop(224)(image)  
            image = transforms.Normalize([0.485], [0.229])(image) #Imagenet transforms

        # Copy image across three channels
        image = np.concatenate([image, image, image], axis=0)
       
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label#, self.stay_id, self.study_id
    
def load_data_phenotyping(parameters, image_dir, batch_size, transform_flag):
    # Upload splits
    train_ids = load_patient_ids('/data/farah/mimic-iv-data/{}/train_listfile.csv'.format(parameters['task']))
    val_ids = load_patient_ids('/data/farah/mimic-iv-data/{}/val_listfile.csv'.format(parameters['task']))
    test_ids = load_patient_ids('/data/farah/mimic-iv-data/{}/test_listfile.csv'.format(parameters['task']))
    
    metadata = process_metadata('/data/shamoutlab/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'
                                , '/data/farah/mimic-iv-data/root/all_stays.csv', task = parameters['task'])
    
    save_metadata_for_loader(metadata.loc[metadata.subject_id.isin(train_ids)], '/data/farah/mimic-iv-data/{}_img/'.format(parameters['task']), 'train_dataloader.csv', parameters['task'], parameters['labels'])
    save_metadata_for_loader(metadata.loc[metadata.subject_id.isin(val_ids)], '/data/farah/mimic-iv-data/{}_img/'.format(parameters['task']), 'val_dataloader.csv', parameters['task'], parameters['labels'])
    save_metadata_for_loader(metadata.loc[metadata.subject_id.isin(test_ids)], '/data/farah/mimic-iv-data/{}_img/'.format(parameters['task']), 'test_dataloader.csv', parameters['task'], parameters['labels'])


    training_data = MIMICCXR('/data/farah/mimic-iv-data/{}_img/train_dataloader.csv'.format(parameters['task']), image_dir, task=parameters['task'], labels=parameters['labels'], transform_flag = transform_flag)
    # No transformations on validation set
    validation_data = MIMICCXR('/data/farah/mimic-iv-data/{}_img/val_dataloader.csv'.format(parameters['task']), image_dir, task=parameters['task'], labels=parameters['labels'], transform_flag = 0)
    # No transformations on test set 
    test_data = MIMICCXR('/data/farah/mimic-iv-data/{}_img/test_dataloader.csv'.format(parameters['task']), image_dir, task=parameters['task'], labels=parameters['labels'], transform_flag = 0)
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return training_data, validation_data, test_data, train_dataloader, val_dataloader, test_dataloader

def labels_phenotyping():
    labels = ['Acute and unspecified renal failure',
                 'Acute cerebrovascular disease',
                 'Acute myocardial infarction',
                 'Cardiac dysrhythmias',
                 'Chronic kidney disease',
                 'Chronic obstructive pulmonary disease and bronchiectasis',
                 'Complications of surgical procedures or medical care',
                 'Conduction disorders',
                 'Congestive heart failure; nonhypertensive',
                 'Coronary atherosclerosis and other heart disease',
                 'Diabetes mellitus with complications',
                 'Diabetes mellitus without complication',
                 'Disorders of lipid metabolism',
                 'Essential hypertension',
                 'Fluid and electrolyte disorders',
                 'Gastrointestinal hemorrhage',
                 'Hypertension with complications and secondary hypertension',
                 'Other liver diseases',
                 'Other lower respiratory disease',
                 'Other upper respiratory disease',
                 'Pleurisy; pneumothorax; pulmonary collapse',
                 'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
                 'Respiratory failure; insufficiency; arrest (adult)',
                 'Septicemia (except in labor)',
                 'Shock',]
    return labels

