import os
from torchvision import datasets
from .augmentation import train_augmentation, test_augmentation, gt_augmentation  
from .mvtecad import MVTecAD, get_df 
from .stats import datasets

def create_dataset(dataset_name:str, datadir:str, class_name:str,
                   img_size:int , mean:list , std:list, aug_info:bool = None,
                    anomaly_ratio:float=0.1, **kwargs):
    trainset, validset, testset  = eval(f"load_{dataset_name}")(
                                                    datadir         = datadir,
                                                    class_name      = class_name,
                                                    img_size        = img_size,
                                                    mean            = mean,
                                                    std             = std,
                                                    aug_info        = aug_info, 
                                                    anomaly_ratio   = anomaly_ratio,
                                                    **kwargs
                                                )
        
    return trainset, validset, testset 

def load_MVTecAD(datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, 
                   anomaly_ratio:float = 0.1, **kwargs):    
    df = get_df(
            datadir       = datadir ,
            class_name    = class_name,
            anomaly_ratio = anomaly_ratio
        )

    trainset = MVTecAD(
                df           = df,
                train_mode   = 'train',
                transform    = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size),
                gt           = True 
            )

    validset = MVTecAD(
                df           = df,
                train_mode   = 'valid',
                transform    = test_augmentation(img_size = img_size, mean = mean, std = std),
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
    
    return trainset, validset, testset 
        