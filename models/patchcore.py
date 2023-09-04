import torch 
import torch.nn as nn 
from anomalib.models.patchcore.torch_model import PatchcoreModel

class PatchCore(nn.Module):
    def __init__(self, model_name:str,  input_size:int, layers:list, num_neighbors:int, coreset_sampling_ratio:float):
        super(PatchCore, self).__init__()
        self.model = PatchcoreModel(
                input_size    = input_size,
                layers        = layers,
                backbone      = model_name,
                pre_trained   = True,
                num_neighbors = num_neighbors
                )
        self.model.feature_extractor.eval()
        
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings = [] 
        
    def train(self):
        self.model.train()
        
    def eval(self, device='cpu'):
        self.model.training = False 
        
        embeddings = torch.vstack(self.embeddings)
        self.model.subsample_embedding(embeddings.to(device), self.coreset_sampling_ratio)
        
    def forward(self, x):
        self.train()
        embedding = self.model(x)
        self.embeddings.append(embedding.detach().cpu())  
        return torch.Tensor([0]) 
    
    def get_score_map(self, x):
        score_map, score = self.model(x)
        return score, score_map