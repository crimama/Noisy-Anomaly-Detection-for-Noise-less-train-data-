from torch.utils.data import Dataset
from datasets.stats import  datasets
import numpy as np 
import os 

class AnomalyDataset(Dataset):
    def __init__(self, normal_dataset:str, anomaly_dataset:str, datadir:str, transforms,
                 anomaly_ratio:float=0.1, total_size:int=10000, train:bool = True):
        super(AnomalyDataset,self).__init__()
        
        self.total_size = total_size
        # Normal/Anomaly size 
        self.normal_size = int(total_size*(1-anomaly_ratio))
        self.anomaly_size = total_size - self.normal_size
        
        # raw Dataset 
        normal = self._dataset_split(
            dataset = self._get_dataset(normal_dataset, datadir, train),
            ratio   = self.normal_size
            )
        anomaly = self._dataset_split(
            dataset = self._get_dataset(anomaly_dataset, datadir, train),
            ratio   = self.anomaly_size
            )
        
        # dataset
        self.data = np.concatenate(
            [normal.data, anomaly.data]
        )
        
        self.label = np.concatenate(
            [np.zeros(len(normal)),np.ones(len(anomaly))]
            )
        
        # transform 
        self.transforms = transforms 
        
    def __len__(self):
        return len(self.data)
                
    def _get_dataset(self, dataset_name:str, datadir:str, train:bool) -> Dataset:
        try:
            dataset = __import__('torchvision').__dict__['datasets'].__dict__[dataset_name](
                root      = os.path.join(datadir, dataset_name),
                train     = train,
                download  = True,
                transform = None
                )
        except:
            dataset = __import__('torchvision').__dict__['datasets'].__dict__[dataset_name](
                root      = os.path.join(datadir, dataset_name),
                split     = 'train' if train else 'test' ,
                download  = True,
                transform = None 
            )
        return dataset 
    
    def _dataset_split(self, dataset:Dataset, ratio:float) -> Dataset:
        
        data_index = np.random.choice(np.arange(len(dataset)), ratio)
        dataset.data = dataset.data[data_index]
        
        if dataset.data.shape[1] == 3:
            dataset.data = np.transpose(dataset.data, axes=(0,2,3,1))
        return dataset 
        
    def __getitem__(self,idx):
        # image 
        data = self.data[idx]
        image1 = self.transforms(data)
        image2 = self.transforms(data)
        
        # label (0:normal, 1:anomaly)
        label = self.label[idx]
        
        return [image1, image2], label


        