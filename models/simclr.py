import torch.nn as nn 
from models import create_model 

class ResNetSimCLR(nn.Module):
    def __init__(self, modelname:str, img_size:int, pretrained:bool = False, out_dim:int =128):
        super(ResNetSimCLR,self).__init__()
        self.backbone = create_model(
                                        modelname   = modelname,
                                        num_classes = out_dim,
                                        img_size    = img_size,
                                        pretrained  = pretrained
                                    )
        dim_mlp = self.backbone.linear.in_features
        
        self.backbone.linear = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp,out_dim)
            )
        
    def forward(self, x):
        return self.backbone(x)
        