import os
from torchvision import datasets
from .augmentation import train_augmentation, test_augmentation, gt_augmentation  
from .mvtecad import MVTecAD, get_df 
from .cifar10 import CIFAR10 
from .stats import datasets

def create_dataset(dataset_name:str, datadir:str, class_name:str,
                   img_size:int , mean:list , std:list, aug_info:bool = None,
                    **params):
    trainset, testset  = eval(f"load_{dataset_name}")(
                                                    datadir         = datadir,
                                                    class_name      = class_name,
                                                    img_size        = img_size,
                                                    mean            = mean,
                                                    std             = std,
                                                    aug_info        = aug_info, 
                                                    **params
                                                )
        
    return trainset, testset 

def load_MVTecAD(datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, baseline: bool = False, anomaly_ratio: float = 0.0):    
    df = get_df(
            datadir       = datadir,
            class_name    = class_name,
            baseline      = baseline,
            anomaly_ratio = anomaly_ratio
        )

    trainset = MVTecAD(
                df           = df,
                train_mode   = 'train',
                transform    = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size),
                gt           = True 
            )

    testset = MVTecAD(
                df           = df,
                train_mode   = 'test',
                transform    = test_augmentation(img_size = img_size, mean = mean, std = std),
                gt_transform = gt_augmentation(img_size = img_size),
                gt           = True 
            )
    
    return trainset, testset 
        
        
def load_CIFAR10(datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, anomaly_ratio: float = 0.0, baseline: bool = True):    
    
    class_name = class_to_idx(class_name) # class_name to idx
    
    trainset = CIFAR10(
        root             = datadir,
        class_name       = class_name,
        anomaly_ratio    = anomaly_ratio,
        baseline         = baseline,
        train            = True,
        transform        = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
        target_transform = None,
        download         = True,
    )
    
    testset = CIFAR10(
        root             = datadir,
        class_name       = class_name,
        anomaly_ratio    = anomaly_ratio,
        baseline         = baseline,
        train            = False,
        transform        = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
        target_transform = None,
        download         = True,
    )
    return trainset, testset 


def class_to_idx(class_name):
    class_to_idx = {'airplane': 0,
                    'automobile': 1,
                    'bird': 2,
                    'cat': 3,
                    'deer': 4,
                    'dog': 5,
                    'frog': 6,
                    'horse': 7,
                    'ship': 8,
                    'truck': 9
                    }
    
    return class_to_idx[class_name]