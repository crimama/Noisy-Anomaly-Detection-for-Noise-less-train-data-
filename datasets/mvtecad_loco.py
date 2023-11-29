from glob import glob 
import os 
import numpy as np 
import pandas as pd 
import cv2 
import torch 
from torch.utils.data import Dataset

from .augmentation import train_augmentation, test_augmentation, gt_augmentation  

class MVTecLoco(Dataset):
    '''
    Example 
        df = get_df(
            datadir       = datadir ,
            class_name    = class_name,
            anomaly_ratio = anomaly_ratio
        )
        trainset = MVTecAD(
            df           = df,
            train_mode   = 'train',
            transform    = train_augmentation,
            gt_transform = gt_augmentation,
            gt           = True 
        )
    '''
    def __init__(self, df: pd.DataFrame, train_mode:str, transform, gt_transform, gt=True):
        '''
        train_mode = ['train','valid','test']
        '''
        self.df = df 
        
        # train / test split 
        self.img_dirs = self.df[self.df['train/test'] == train_mode][0].values # column 0 : img_dirs 
        self.labels = self.df[self.df['train/test'] == train_mode]['anomaly'].values 
        
        # ground truth 
        self.gt = gt # mode 
        self.gt_transform = gt_transform 

        # Image 
        self.transform = transform 
        
        self.name = 'MVTecAD'
        
    def _get_ground_truth(self, img_dir, img):
        img_dir = img_dir.split('/')
        if img_dir[-2] !='good':
            img_dir[-3] = 'ground_truth'
            img_dir[-1] = img_dir[-1].split('.')[0]
            img_dir.append('000.png')
            img_dir = '/'.join(img_dir)
            image = cv2.imread(img_dir)
        else:
            image = np.zeros_like(torch.permute(img,dims=(1,2,0))).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image 
        
    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self,idx):
        img_dir = self.img_dirs[idx]
        img = cv2.imread(img_dir)
        img = self.transform(img)
        img = img.type(torch.float32)
        label = self.labels[idx]
        
        if self.gt:
            gt = self._get_ground_truth(img_dir,img)
            gt = self.gt_transform(gt)
            gt = (gt > 0).float()
            
            return img, label, gt, idx
        
        else:
            return img, label 
        
      
                