import os
from torchvision import datasets
from .augmentation import train_augmentation, test_augmentation
from .build import AnomalyDataset
from .stats import datasets

def create_dataset(normal_dataset:str, anomaly_dataset:str, datadir:str,
                   img_size:int , mean:list , std:list, aug_info:bool = None,
                    anomaly_ratio:float=0.1, total_size:int=10000):
    
    trainset = AnomalyDataset(
        normal_dataset  = normal_dataset,
        anomaly_dataset = anomaly_dataset,
        datadir         = datadir,
        transforms      = train_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info),
        anomaly_ratio   = anomaly_ratio,
        total_size      = total_size,
        train           = True 
        )
    testset = AnomalyDataset(
        normal_dataset  = normal_dataset,
        anomaly_dataset = anomaly_dataset,
        datadir         = datadir,
        transforms      = test_augmentation(img_size=img_size, mean=mean, std=std),
        anomaly_ratio   = anomaly_ratio,
        total_size      = 1000,
        train           = False 
        )
    
    return trainset, testset 