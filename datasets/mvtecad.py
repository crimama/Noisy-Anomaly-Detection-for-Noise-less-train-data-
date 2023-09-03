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

        self.transform = transform 
        
    def _get_ground_truth(self, img_dir,img):
        img_dir = img_dir.split('/')
        if img_dir[-2] !='good':
            img_dir[-3] = 'ground_truth'
            img_dir[-1] = img_dir[-1].split('.')[0] + '_mask.png'
            img_dir = '/'.join(img_dir)
            image = cv2.imread(img_dir)
        else:
            image = np.zeros_like(torch.permute(img,dims=(1,2,0)))
            
        return image 
        
    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self,idx):
        img_dir = self.img_dirs[idx]
        img = cv2.imread(img_dir)
        img = self.transform(img)
        
        label = self.labels[idx]
        
        if self.gt:
            gt = self._get_ground_truth(img_dir,img)
            gt = self.gt_transform(gt)
            gt = (gt >= 0.5).float()
            
            return img, label, gt 
        
        else:
            return img, label 
        
        

def get_df(datadir: str, class_name: str, anomaly_ratio: float, method, **params: dict):
    '''
    args:
        normal_ratio : 0 ~ 0.25 for PatchCore 
    Example:
        df = get_df(
                datadir       = datadir ,
                class_name    = class_name,
                anomaly_ratio = anomaly_ratio
            )
    '''
    
    # get img_dirs dataframe 
    img_dirs = get_img_dirs(datadir=datadir, class_name=class_name)
    
    # train test split
    df = train_test_split(                
        img_dirs            = img_dirs,
        train_anomaly_ratio = anomaly_ratio
        )
    
    if method == 'PatchCore':
        length_train = len(df[df['train/test'] == 'train'])
        length_test = int(length_train * (1-params['normal_ratio']))
        index_test = df[df['train/test'] == 'train'].sample(length_test).index
        df.loc[index_test,'train/test'] = 'test'
        
    elif method == 'STPM':
        # train valid split, valid is 20% of train
        length_train = len(df[df['train/test'] == 'train'])
        length_valid = int(length_train*0.2)
        index_valid = df[df['train/test'] == 'train'].sample(length_valid).index
        df.loc[index_valid,'train/test'] = 'valid'
    
    return df 

    

def get_img_dirs(datadir:str, class_name:str) -> pd.DataFrame:
    class_name = '*' if class_name =='all' else class_name 
    
    img_dirs = pd.Series(sorted(glob(os.path.join(datadir,'MVTecAD', class_name,'*/*/*.png'))))
    img_dirs = pd.DataFrame(img_dirs[img_dirs.apply(lambda x : x.split('/')[-3]) != 'ground_truth']).reset_index(drop=True)
    img_dirs['anomaly'] = img_dirs[0].apply(lambda x : 1 if x.split('/')[-2] != 'good' else 0)
    img_dirs['train/test'] = ''
    return img_dirs 


def train_test_split(img_dirs:pd.DataFrame, train_anomaly_ratio:float) -> pd.DataFrame:
    img_dirs['train/test'] = img_dirs[0].apply(lambda x : x.split('/')[-3]) # allocate initial train/test label 
    if train_anomaly_ratio == 0:
        pass 
    else:
        # compute the number of data for each label initially
        n_train =  len(img_dirs[img_dirs['train/test'] == 'train']) 
        
        # compute the number of data swaping btw train and test 
        n_train_anomaly = int(n_train * train_anomaly_ratio)
        i_train_anomaly = img_dirs[(img_dirs['train/test'] == 'test') & (img_dirs['anomaly'] == 1)].sample(n_train_anomaly).index # test anomaly -> train 
        i_test_normal = img_dirs[(img_dirs['train/test'] == 'train') & (img_dirs['anomaly'] == 0)].sample(n_train_anomaly).index # train normal -> test 
        
        img_dirs.loc[i_train_anomaly,'train/test'] = 'train'
        img_dirs.loc[i_test_normal, 'train/test'] = 'test'
    
    return img_dirs 


        