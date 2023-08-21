import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class MixLoss(nn.Module):
    def __init__(self, **criterion_params):
        super(MixLoss,self).__init__()          
        
        self.feat_criterion = __import__('criterions').__dict__['FeatDist'](
            layers = criterion_params['layers']
        )
        self.cont_criterion = __import__('criterions').__dict__['InfoNCE'](
            temperature = criterion_params['temperature']
        )
        self.alpha = criterion_params['alpha']
        
    def __call__(self, outputs:list):
        [_, _, x_t, x_s] = outputs
        
        feat_loss = self.feat_criterion(outputs)
        cont_loss = self.cont_criterion([x_t, x_s])
        
        total_loss = (1-self.alpha) * feat_loss + self.alpha * cont_loss 
        
        return total_loss 