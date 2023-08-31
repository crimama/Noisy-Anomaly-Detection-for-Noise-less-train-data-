from glob import glob 
import os 

import numpy as np 
import pandas as pd 

import cv2 

import torch 
from torch.utils.data import Dataset

class MVTecAD(Dataset):
    '''
    Example 
        trainset = MVTecAD(
            datadir       = cfg.DATASET.datadir,
            class_name    = cfg.DATASET.params.class_name,
            transform     = train_augmentation(img_size = cfg.DATASET.img_size, mean = cfg.DATASET.mean, std = cfg.DATASET.std, aug_info = cfg.DATASET.aug_info),
            gt_transform  = gt_augmentation(img_size = cfg.DATASET.img_size),
            anomaly_ratio = cfg.DATASET.anomaly_ratio,
            mode          = 'train',
            gt            = True 
        )
        testset = MVTecAD(
            datadir       = cfg.DATASET.datadir,
            class_name    = cfg.DATASET.params.class_name,
            transform     = test_augmentation(img_size = cfg.DATASET.img_size, mean = cfg.DATASET.mean, std = cfg.DATASET.std),
            gt_transform  = gt_augmentation(img_size = cfg.DATASET.img_size),
            anomaly_ratio = cfg.DATASET.anomaly_ratio,
            mode          = 'test',
            df            = trainset.df,
            gt            = True 
        )
    '''
    def __init__(self, datadir:str , class_name:str, transform, gt_transform,
                anomaly_ratio:float, train_mode='train', gt=True, df=None):
        
        if df is not None:
            self.df = df 
        else: 
            # init image dirs & labels 
            self.df = _train_test_split(                
                img_paths = _get_img_dirs(datadir=datadir, class_name=class_name),
                train_anomaly_ratio = anomaly_ratio
            )
        
        # train / test split 
        self.img_dirs = self.df[self.df['train/test'] == train_mode][0].values # column 0 : img_dirs 
        self.labels = self.df[self.df['train/test'] == train_mode]['anomaly'].values 
        
        # ground truth 
        self.gt = gt # mode 
        self.gt_transform = gt_transform 

        self.transform = transform 
        
    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self,idx):
        img_dir = self.img_dirs[idx]
        img = cv2.imread(img_dir)
        img = self.transform(img)
        
        label = self.labels[idx]
        
        if self.gt:
            gt = _get_ground_truth(img_dir,img)
            gt = self.gt_transform(gt)
            gt = (gt >= 0.5).float()
            
            return img, label, gt 
        
        else:
            return img, label 
        
        

def _get_ground_truth(img_dir,img):
    img_dir = img_dir.split('/')
    if img_dir[-2] !='good':
        img_dir[-3] = 'ground_truth'
        img_dir[-1] = img_dir[-1].split('.')[0] + '_mask.png'
        img_dir = '/'.join(img_dir)
        image = cv2.imread(img_dir)
    else:
        image = np.zeros_like(torch.permute(img,dims=(1,2,0)))
        
    return image 
    

def _get_img_dirs(datadir, class_name):
        class_name = '*' if class_name =='all' else class_name 
        
        img_paths = pd.Series(sorted(glob(os.path.join(datadir,'MVTecAD', class_name,'*/*/*.png'))))
        img_paths = pd.DataFrame(img_paths[img_paths.apply(lambda x : x.split('/')[-3]) != 'ground_truth']).reset_index(drop=True)
        img_paths['anomaly'] = img_paths[0].apply(lambda x : 1 if x.split('/')[-2] != 'good' else 0)
        img_paths['train/test'] = ''
        return img_paths 


def _train_test_split(img_paths, train_anomaly_ratio):
    if train_anomaly_ratio == 0:
        img_paths['train/test'] = img_paths[0].apply(lambda x : x.split('/')[-3])
    else:
        anomaly_index = list(img_paths[img_paths['anomaly']==1].index)
        normal_index = list(img_paths[img_paths['anomaly']==0].index)

        # Anomaly : test -> train 
        anomaly_train = int(len(normal_index) * train_anomaly_ratio)
        anomaly_train_index = np.random.choice(anomaly_index, anomaly_train,replace=False)
        img_paths.loc[anomaly_train_index,'train/test'] = 'train'

        # Anomaly test 
        anomaly_test_index = np.array(list(set(anomaly_index) - set(anomaly_train_index)))
        img_paths.loc[anomaly_test_index, 'train/test'] = 'test'

        # Normal : train -> test 
        normal_test_index = np.random.choice(normal_index, anomaly_train,replace=False)
        img_paths.loc[normal_test_index, 'train/test'] = 'test'
        # normal : train 

        normal_train_index = np.array(list(set(normal_index) - set(normal_test_index)))
        img_paths.loc[normal_train_index, 'train/test'] = 'train'
    
    return img_paths 


        