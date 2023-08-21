import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class FeatDist(nn.Module):
    def __init__(self, layers:list = [0,1,2,3]):
        super(FeatDist,self).__init__()
        self.layers = layers 
        
        
    def __call__(self, outputs):
        '''
        outputs = [t_features, s_features, x_t, x_s]
        '''
        t_outputs = outputs[0]
        s_outputs = outputs[1]
        
        t_outputs = [t_outputs[x] for x in self.layers]
        s_outputs = [s_outputs[x] for x in self.layers]
        
        total_loss = 0 
        for t,s in zip(t_outputs, s_outputs):
            t,s = F.normalize(t, dim=1), F.normalize(s, dim=1)
            total_loss += torch.sum((t.type(torch.float32) - s.type(torch.float32)) ** 2, 1).mean()
        return total_loss 