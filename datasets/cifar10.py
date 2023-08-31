import os 
import numpy as np 
from torch.utils.data import Dataset 
from torchvision import datasets 

class CIFAR10(Dataset):
    def __init__(self, normal_label: int, datadir: str, transform,  anomaly_ratio: float = 0, train_mode: bool = False):
        super(Dataset,self).__init__() 
        self.normal_label = normal_label 
        
        if train_mode:
            self.data, self.targets = self._get_anomaly(
                dataset       = datasets.CIFAR10(root = os.path.join(datadir,'CIFAR10'), train = train_mode, download  = True, transform = None),
                anomaly_ratio = anomaly_ratio
            )
        else:
            dataset = datasets.CIFAR10(root = os.path.join(datadir,'CIFAR10'), train = train_mode, download  = True, transform = None)
            self.data, self.targets = dataset.data, dataset.targets 
            
        self.transform = transform 
        
    def __len__(self):
        return len(self.dataset)
    
    def _get_anomaly(self, dataset, anomaly_ratio):
        # Get Normal/Anomaly Index 
        normal_index = np.where(np.array(dataset.targets) == 0 )[0]
        anomaly_index = np.where(np.array(dataset.targets) != 0 )[0]

        # Get the number of Normal/Anomaly required
        use_index = [] 
        num_anomaly = int(len(normal_index) * anomaly_ratio)
        num_normal = len(normal_index) - num_anomaly # Swap btw Anomaly - Normal 
        
        # Get index for training 
        use_index.append(np.random.choice(normal_index, num_normal, replace=False))
        use_index.append(np.random.choice(anomaly_index, num_anomaly, replace=False))
        use_index = np.concatenate(use_index).astype(int)
        return dataset.data[use_index], np.array(dataset.targets)[use_index]
    
    def __getitem__(self, idx): 
        Image, Label = self.data[idx], self.targets[idx]
        
        # preprocess 
        Image = self.transform(Image)
        Label = 0 if Label == self.normal_label else 1 
        
        return Image, Label 