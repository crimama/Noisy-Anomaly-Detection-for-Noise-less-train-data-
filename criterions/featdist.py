import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class FeatDist(nn.Module):
    def __init__(self, layers:list = [0,1,2,3]):
        super(FeatDist,self).__init__()
        self.layers = layers 
        
        
    def __call__(self, t_outputs, s_outputs):
        t_outputs = [t_outputs[x] for x in self.layers]
        s_outputs = [s_outputs[x] for x in self.layers]
        
        total_loss = 0 
        for t,s in zip(t_outputs, s_outputs):
            t,s = F.normalize(t, dim=1), F.normalize(s, dim=1)
            total_loss += torch.sum((t.type(torch.float32) - s.type(torch.float32)) ** 2, 1).mean()
        return total_loss 