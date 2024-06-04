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
import os


class MIMICCXR(Dataset):
    def __init__(self, paths, args, transform=None, split='train'):
        self.data_dir = args.cxr_data_root
        self.args = args
        self.CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']
        self.filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}

        metadata = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-metadata.csv')
        labels = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-chexpert.csv')
        labels[self.CLASSES] = labels[self.CLASSES].fillna(0)
        labels = labels.replace(-1.0, 0.0)
        
        splits = pd.read_csv(f'{self.data_dir}/mimic-cxr-ehr-split.csv')


        metadata_with_labels = metadata.merge(labels[self.CLASSES+['study_id'] ], how='inner', on='study_id')


        self.filesnames_to_labels = dict(zip(metadata_with_labels['dicom_id'].values, metadata_with_labels[self.CLASSES].values))
        self.filenames_loaded = splits.loc[splits.split==split]['dicom_id'].values
        self.transform = transform
        self.filenames_loaded = [filename  for filename in self.filenames_loaded if filename in self.filesnames_to_labels]

    def __getitem__(self, index):
        if isinstance(index, str):
            img = Image.open(self.filenames_to_path[index]).convert('RGB')
            #print(self.filenames_to_path[index])
            labels = torch.tensor(self.filesnames_to_labels[index]).float()
            if self.transform is not None:
                img = self.transform(img)
            return img, labels
          
        
        filename = self.filenames_loaded[index]
        img = Image.open(self.filenames_to_path[filename]).convert('RGB')
        labels = torch.tensor(self.filesnames_to_labels[filename]).float()

        if self.transform is not None:
            img = self.transform(img)
        return img, labels
    
    def __len__(self):
        return len(self.filenames_loaded)


def get_transforms(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transforms = []
    train_transforms.append(transforms.Resize(args.resize))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    train_transforms.append(transforms.CenterCrop(224))#args.crop))
    train_transforms.append(transforms.ToTensor())
    #train_transforms.append(normalize)      


    test_transforms = []
    test_transforms.append(transforms.Resize(args.resize))
    test_transforms.append(transforms.CenterCrop(224))#args.crop))
    test_transforms.append(transforms.ToTensor())
    #test_transforms.append(normalize)

    return train_transforms, test_transforms


class Clip(object):
    """Transformation to clip image values between 0 and 1."""

    def __call__(self, sample):
        return torch.clip(sample, 0, 1)

    

class RandomCrop(object):
    "Randomly crop an image"
    
    def __call__(self, sample):
        resize = 256
        #print(np.random.uniform(0.4*resize,resize,1))
        random_crop_size = int(np.random.uniform(0.6*resize,resize,1))
        sample=transforms.RandomCrop(random_crop_size)(sample)
        return sample
    
    
class RandomColorDistortion(object):
    "Apply random color distortions to the image"
    
    def __call__(self, sample):
        resize=256

        # Random color distortion
        strength = 1.0 # 1.0 imagenet setting and CIFAR uses 0.5
        brightness = 0.8 * strength 
        contrast = 0.8 * strength
        saturation = 0.8 * strength
        hue = 0.2 * strength
        prob = np.random.uniform(0,1,1) 
        if prob < 0.8:
            sample=transforms.ColorJitter(brightness, contrast, saturation, hue)(sample)

        # Random Grayscale
        sample=transforms.RandomGrayscale(p=0.2)(sample)

        # Gaussian blur also based on imagenet but not used for CIFAR
        #prob = np.random.uniform(0,1,1)
        #if prob < 0.3:
        #    sample=transforms.GaussianBlur(kernel_size=resize//10)(sample)
        #    sample=transforms.Pad(0)(sample)
        return sample 
    

def get_transforms_simclr(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transforms = []
    # Resize all images to same size, then randomly crop and resize again
    train_transforms.append(transforms.Resize([args.resize, args.resize]))
    # Random affine
    train_transforms.append(transforms.RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25)))
    # Random crop
    train_transforms.append(RandomCrop())
    # Resize again
    # train_transforms.append(transforms.Resize([args.resize, args.resize], interpolation=3))
    train_transforms.append(transforms.Resize([224, 224], interpolation=3))
    # Random horizontal flip 
    train_transforms.append(transforms.RandomHorizontalFlip())
    # Random color distortions
    train_transforms.append(RandomColorDistortion())
    # Convert to tensor
    train_transforms.append(transforms.ToTensor())
    # Clip values between 0 and 1 and normalize
    #train_transforms.append(Clip())
    #train_transforms.append(normalize)      

    test_transforms = []
    # Resize all images to same size, then center crop and resize again
    test_transforms.append(transforms.Resize([args.resize, args.resize]))
    crop_proportion=0.875
    test_transforms.append(transforms.CenterCrop([int(0.875*args.resize), int(0.875*args.resize)]))
    # test_transforms.append(transforms.Resize([args.resize, args.resize], interpolation=3))
    test_transforms.append(transforms.Resize([224, 224], interpolation=3))
    #Convert to tensor
    test_transforms.append(transforms.ToTensor())
    # Clip values between 0 and 1 and normalize
    #test_transforms.append(Clip())
    #test_transforms.append(normalize)

    return train_transforms, test_transforms


# Note this function needs to be editted to mimic function above 
def visualize_transforms_simclr(args, orig_img, split='train'):
    # Create array of images 
    print(orig_img)
    new_images = [orig_img]
    tt = ['Original image']
    #normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if split == 'train':
        # Resize all images to same size
        new_images = new_images + [transforms.Resize([args.resize, args.resize])(orig_img)]
        tt = tt + ['Resize original image']
        # Random affine
        new_images = new_images + [transforms.RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25))(new_images[-1])]
        tt = tt + ['Random affine']
        # Random crop
        new_images = new_images + [RandomCrop()(new_images[-1])]
        tt = tt + ['Random Crop']
        # Resize to 256 x 256
        new_images = new_images + [transforms.Resize([args.resize, args.resize], interpolation=3)(new_images[-1])]
        tt = tt + ['Resize patch']
        # Random horizontal flip 
        new_images = new_images + [transforms.RandomHorizontalFlip()(new_images[-1])]
        tt = tt + ['Random horizontal flip']
        # Random color distortions
        new_images = new_images + [RandomColorDistortion()(new_images[-1])]
        tt = tt + ['Random color distortion']
        
        # Convert all to tensors
        for i in range(0, len(new_images)):
            new_images[i]=transforms.ToTensor()(new_images[i])
#         # Clip values between 0 and 1 and normalize
#         new_images = new_images + [Clip()(new_images[-1])]
#         tt = tt + ['Clip values (0,1)']
#         # Normalize values
#         new_images = new_images + [normalize(new_images[-1])]
#         tt = tt + ['Normalize values']
    return new_images, tt




def get_cxr_datasets(args):
    if args.transforms_cxr=='simclrv2':
        print("Appling SimCLR image transforms...")
        train_transforms, test_transforms = get_transforms_simclr(args)
    else:
        print("Applying linear evaluation transforms...")
        train_transforms, test_transforms = get_transforms(args)

    data_dir = args.cxr_data_root
    filepath = f'{args.cxr_data_root}/new_paths.npy'
    if os.path.exists(filepath):
        paths = np.load(filepath)
    else:
        paths = glob.glob(f'{data_dir}/resized/**/*.jpg', recursive = True)
        np.save(filepath, paths)
    
    dataset_train = MIMICCXR(paths, args, split='train', transform=transforms.Compose(train_transforms))
    dataset_validate = MIMICCXR(paths, args, split='validate', transform=transforms.Compose(test_transforms),)
    dataset_test = MIMICCXR(paths, args, split='test', transform=transforms.Compose(test_transforms),)

    return dataset_train, dataset_validate, dataset_test

