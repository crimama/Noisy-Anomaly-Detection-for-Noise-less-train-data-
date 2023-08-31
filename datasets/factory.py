import os
from torchvision import datasets
from .augmentation import train_augmentation, test_augmentation, gt_augmentation  
from .mvtecad import MVTecAD
from .stats import datasets

def create_dataset(dataset_name:str, datadir:str, class_name:str,
                   img_size:int , mean:list , std:list, aug_info:bool = None,
                    anomaly_ratio:float=0.1, **kwargs):
    trainset, testset  = eval(f"load_{dataset_name}")(
                                                    datadir         = datadir,
                                                    class_name      = class_name,
                                                    img_size        = img_size,
                                                    mean            = mean,
                                                    std             = std,
                                                    aug_info        = aug_info, 
                                                    anomaly_ratio   = anomaly_ratio,
                                                    **kwargs
                                                )
        
    return trainset, testset 

def load_MVTecAD(datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, 
                   anomaly_ratio:float = 0.1, **kwargs):
    trainset = MVTecAD(
        datadir       = datadir,
        class_name    = class_name,
        transform     = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
        gt_transform  = gt_augmentation(img_size = img_size),
        anomaly_ratio = anomaly_ratio,
        train_mode    = 'train'
        )
    
    testset = MVTecAD(
        datadir       = datadir,
        class_name    = class_name,
        transform     = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
        gt_transform  = gt_augmentation(img_size = img_size),
        anomaly_ratio = anomaly_ratio,
        train_mode    = 'test',
        df            = trainset.df
        )
    
    return trainset, testset 
        