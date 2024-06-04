import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class EHRdataset(Dataset):
    def __init__(self, args, discretizer, normalizer, listfile, dataset_dir, return_names=True, period_length=48.0, transforms=None):
        self.return_names = return_names
        self.discretizer = discretizer
        self.normalizer = normalizer
        self._period_length = period_length

        self._dataset_dir = dataset_dir
        listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES = self._listfile_header.strip().split(',')[3:]
        self._data = self._data[1:]
        self.transforms = transforms


        self._data = [line.split(',') for line in self._data]
        self.data_map = {
            mas[0]: {
                'labels': list(map(int, mas[3:])),
                'stay_id': float(mas[2]),
                'time': float(mas[1]),
                }
                for mas in self._data
        }

        self.names = list(self.data_map.keys())
    
#     def _read_timeseries(self, ts_filename):
        
#         ret = []
#         with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
#             header = tsfile.readline().strip().split(',')
#             assert header[0] == "Hours"
#             for line in tsfile:
#                 mas = line.strip().split(',')
#                 ret.append(np.array(mas))
#         return (np.stack(ret), header)

    def read_timeseries(self,ts_filename, lower_bound=0,upper_bound=12):
        
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
            
                t = float(mas[0])
                if t < lower_bound:
                    continue
                elif (t> lower_bound) & (t <upper_bound) :
                    ret.append(np.array(mas))
                elif t > upper_bound:
                    break
            
    #             if time_bound is not None:
    #                 t = float(mas[0])
    #                 if t > time_bound + 1e-6:
    #                     break
    #             ret.append(np.array(mas))
        try: 
            return (np.stack(ret), header)
        except ValueError:
            
            ret = ([['0.11666666666666667', '', '', '', '', '', '', '', '', '109', '',
                     '', '', '30', '', '', '', ''],
                    ['0.16666666666666666', '', '61.0', '', '', '', '', '', '', '109',
                    '', '64', '97.0', '29', '74.0', '', '', '']])
            return (np.stack(ret), header)
            
         
    
#     def read_by_file_name(self, index):
#         t = self.data_map[index]['time']
#         y = self.data_map[index]['labels']
#         stay_id = self.data_map[index]['stay_id']
#         (X, header) = self._read_timeseries(index)
#         print(index)
#         return {"X": X,
#                 "t": t,
#                 "y": y,
#                 'stay_id': stay_id,
#                 "header": header,
#                 "name": index}

    def read_by_file_name(self,index, lower_bound=0,upper_bound=12):
        t = self.data_map[index]['time'] #if upper_bound is None else upper_bound
        y = self.data_map[index]['labels']
        stay_id = self.data_map[index]['stay_id']
        (X, header) = self.read_timeseries(index, lower_bound=lower_bound,upper_bound=upper_bound)
        
        return {"X": X,
                "t": t,
                "y": y,
                'stay_id': stay_id,
                "header": header,
                "name": index}

    def __getitem__(self, index,lower,upper):
        if isinstance(index, int):
            index = self.names[index]
        ret = self.read_by_file_name(index,lower,upper)
        data = ret["X"]
#         print(index)
        ts = data.shape[0]#ret["t"] if ret['t'] > 0.0 else self._period_length
        
        
        ## Added block
        if self.transforms is not None:
            data = self.transforms(data)
            
            for i in range(len(data)):
                data[i] = self.discretizer.transform(data[i], end=ts)[0]
                print(data[i]).shape
                if 'gaussian' in self.transforms.augmentation and i != 0:
                    data[i] = self.transforms.gaussian_blur(data[i])
                if 'sampling' in self.transforms.augmentation and i != 0: #carry last value forward 
                    data[i] = self.transforms.downsample(data[i])
                if (self.normalizer is not None):
                    data[i] = self.normalizer.transform(data[i])
        else:
            data = self.discretizer.transform(data, end=ts)[0]
            if (self.normalizer is not None):
                data = self.normalizer.transform(data)
        #########  
        
        # data = self.discretizer.transform(data, end=ts)[0] 
        # if (self.normalizer is not None):
        #     data = self.normalizer.transform(data)
        # print(data.shape)

        ys = ret["y"]
        names = ret["name"]
        ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
        stay_ids = ret['stay_id']
        return data, ys

    
    def __len__(self):
        return len(self.names)


def get_datasets(discretizer, normalizer, args):
    # if context == True:
    #     transform = MultiTransform(views=11, normal_values=discretizer._id_normal_values, _is_categorical_channel=discretizer._is_categorical_channel, augmentation=augmentation, begin_pos=begin_pos)
    # else:
    #     transform = None
    # changed definition of normal_values
    # augmentation = 'gaussian'
    # transform = MultiTransform(views=11, normal_values=discretizer._normal_values, _is_categorical_channel=discretizer._is_categorical_channel, augmentation=augmentation, begin_pos=discretizer._start_time)
    transform = None
    train_ds = EHRdataset(args, discretizer, normalizer, f'{args.ehr_data_root}/{args.task}/train_listfile.csv', os.path.join(args.ehr_data_root, f'{args.task}/train'), transforms=transform)
    val_ds = EHRdataset(args, discretizer, normalizer, f'{args.ehr_data_root}/{args.task}/val_listfile.csv', os.path.join(args.ehr_data_root, f'{args.task}/train'), transforms = transform)
    test_ds = EHRdataset(args, discretizer, normalizer, f'{args.ehr_data_root}/{args.task}/test_listfile.csv', os.path.join(args.ehr_data_root, f'{args.task}/test'), transforms = transform)
    return train_ds, val_ds, test_ds

def get_data_loader(discretizer, normalizer, dataset_dir, batch_size):
    train_ds, val_ds, test_ds = get_datasets(discretizer, normalizer, dataset_dir)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16)

    return train_dl, val_dl
        
def my_collate(batch):
    x = [item[0] for item in batch]
    x, seq_length = pad_zeros(x)
    targets = np.array([item[1] for item in batch])
    return [x, targets, seq_length]

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




class MultiTransform(object):

    def __init__(
        self,
        views,
        normal_values,
        _is_categorical_channel,
        augmentation,
        begin_pos
    ):
        self.views = views
        self.normal_values = normal_values
        self.rows = np.array([value for value in self.normal_values.values()])
        self.augmentation = augmentation
        self.continuous_variable = [0 if _is_categorical_channel[key] == True else 1 for key in _is_categorical_channel]
        self.begin_pos = begin_pos
        
    def vertical_mask(self, data, max_percent=0.4):
        # mask over each timestep (t, features)
        length = data.shape[0]
        if length < 4:
            return data
        size = int(np.random.randint(low=0, high=max(int(max_percent*length),1), size=1))
        a = np.zeros(length , dtype=int)
        a[:size] = 1
        np.random.shuffle(a)
        a = a.astype(bool)
        data[a,1:] = self.rows
        return data

    def horizontal_mask(self, data, max_percent=0.4):
        # mask over each feature (t, features)
        length = data.shape[1] - 1
        size = int(np.random.randint(low=0, high=max(int(max_percent*length),1), size=1))
        features = np.unique(np.random.randint(low=1, high=length, size=size))
        for i in features:
            data[:,i+1] = self.normal_values[i]
        return data
    
    def drop_start(self, data, max_percent=0.4):
        length = data.shape[0]
        start = int(np.random.randint(low=0, high=max(int(max_percent*length),1), size=1))
        return data[start:,:]

    def gaussian_blur(self, data):
        mean, std = 1,0 
        data[:, self.begin_pos] = data[:, self.begin_pos]  + np.random.normal(mean, std, (data.shape[0], len(self.begin_pos)))
        return data

    def rotation(self, data):
        if choice([0,1]):
            return np.flip(data, axis=0)
        return data

    def downsample(self, data):
        if data.shape[0] < 20:
            return data
        step = choice([1, 2, 3])
        return data[::step]

    def __call__(self, data):
        data_views = []                    
        data_views.append(self.vertical_mask(data))
        data_views.append(self.horizontal_mask(data))
        data_views.append(self.horizontal_mask(self.vertical_mask(data)))
        data_views.append((self.drop_start(data)))
        data_views.append(data)

        return data_views
    
    
def get_transforms(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transforms = []
    train_transforms.append(transforms.Resize(args.resize))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    train_transforms.append(transforms.CenterCrop(args.crop))
    train_transforms.append(transforms.ToTensor())
    #train_transforms.append(normalize)      


    test_transforms = []
    test_transforms.append(transforms.Resize(args.resize))
    test_transforms.append(transforms.CenterCrop(args.crop))
    test_transforms.append(transforms.ToTensor())
    #test_transforms.append(normalize)

    return train_transforms, test_transforms