import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from info_nce import InfoNCE as _InfoNCE

class InfoNCE(nn.Module):
    def __init__(self, temperature:float = 0.5):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.criterion = _InfoNCE()

    def forward(self, features:list):
        query = features[0]
        positive_key = features[1]
        
        loss = self.criterion(query, positive_key)
        
        return loss