import timm 
import torch 
from torch import Tensor, nn
import torch.nn.functional as F 

from anomalib.models.stfpm.torch_model import STFPMModel
from anomalib.models.stfpm.loss import STFPMLoss

class AnomalibSTPM(nn.Module):
    def __init__(self, model_name:str,  input_size:list, layers:list):
        super(AnomalibSTPM,self).__init__()
        self.model = STFPMModel(
            layers     = layers,
            input_size = input_size,
            backbone   = model_name
        )

        self.criterion = STFPMLoss() 
        
    
    def _forward(self, x):
        if self.model.tiler:
            x = self.model.tiler.tile(x)
            
        t_features = self.model.teacher_model(x)
        s_features = self.model.student_model(x)
        
        return t_features, s_features 
        

    def forward(self, x, only_loss:bool = True) -> Tensor:
        # Inference 
        t_features, s_features = self._forward(x)
        
        # Comput Loss 
        loss = self.criterion(t_features, s_features)
        
        if only_loss:
            return loss 
        else:
            return loss, [t_features, s_features]

        
    def get_score_map(self, outputs: list) -> torch.Tensor:
        '''
        outputs = [t_outputs, s_outputs]
        '''
        t_features, s_features = outputs[0], outputs[1]
        
        output = self.model.anomaly_map_generator(
            teacher_features = t_features,
            student_features = s_features
        )
        
        if self.model.tiler:
            output = self.mode.tiler.untile(output)
            
        return output 
    
    