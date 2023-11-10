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

        # Image 
        self.transform = transform 
        
        self.name = 'MVTecAD'
        
    def _get_ground_truth(self, img_dir, img):
        img_dir = img_dir.split('/')
        if img_dir[-2] !='good':
            img_dir[-3] = 'ground_truth'
            img_dir[-1] = img_dir[-1].split('.')[0] + '_mask.png'
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
            
            return img, label, gt 
        
        else:
            return img, label 
        
        

def get_df(datadir: str, class_name: str, baseline: bool = True, anomaly_ratio: float = 0.0):
    '''
    args:
        datadir : root of data 
        class_name : the name of category 
        baseline : dataset for reproduce performance of baseline if True or dataset for experiment of fully unsupervised 
    Example:
        df = get_df(
                datadir       = './Data' , 
                class_name    = 'toothbrush'
            ) 
    '''
    # get img_dirs dataframe 
    img_dirs = get_img_dirs(datadir=datadir, class_name=class_name)
    img_dirs['train/test'] = img_dirs[0].apply(lambda x : x.split('/')[-3]) # allocate initial train/test label 
    
    if baseline:
        df = img_dirs
    else:
        # train test split
        df = train_test_split(                
            df      = img_dirs,
            anomaly_ratio = anomaly_ratio
            )
    return df 

    

def get_img_dirs(datadir:str, class_name:str) -> pd.DataFrame:
    '''
        디렉토리 내 이미지 디렉토리들을 가져오는 메소드 
    '''
    class_name = '*' if class_name =='all' else class_name 
    
    img_dirs = pd.Series(sorted(glob(os.path.join(datadir,'MVTecAD', class_name,'*/*/*.png'))))
    img_dirs = pd.DataFrame(img_dirs[img_dirs.apply(lambda x : x.split('/')[-3]) != 'ground_truth']).reset_index(drop=True)
    img_dirs['anomaly'] = img_dirs[0].apply(lambda x : 1 if x.split('/')[-2] != 'good' else 0)
    img_dirs['train/test'] = ''
    return img_dirs 

def train_test_split(df, anomaly_ratio):
    
    num_train = len(df[df['train/test'] == 'train'])
    num_max_anomaly = int(num_train * 0.1)
    num_anomaly_train = int(num_train*anomaly_ratio)

    # test 데이터 중 train으로 사용 될 후보군 sampling 
    unfix_anomaly_index = df[(df['train/test'] == 'test') & (df['anomaly'] == 1)].sample(num_max_anomaly).index
    df.loc[unfix_anomaly_index, 'train/test'] = 'unfix'

    # train으로 사용될 후보군 제외 후 test 셋 고정 
    fix_test = df[(df['train/test'] == 'test')].reset_index(drop=True)

    # train의 anomaly 수 만큼 sampling 후 trainset에서 제외 
    notuse_train_normal = df[df['train/test']=='train'].sample(num_anomaly_train).index
    df.loc[notuse_train_normal,'train/test'] = 'notuse'

    # train 의 anomaly 수 만큼 후보군에서 sampling 및 train에 추가 
    train_anomaly_index = df[df['train/test']=='unfix'].sample(num_anomaly_train).index
    df.loc[train_anomaly_index,'train/test'] = 'train'

    # train 셋 고정 
    fix_train = df[df['train/test']=='train'].reset_index(drop=True)

    final_df = pd.concat([fix_train, fix_test]).reset_index(drop=True)
    
    return final_df 


        