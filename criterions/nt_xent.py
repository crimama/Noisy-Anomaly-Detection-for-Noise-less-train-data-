import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature:float = 0.5):
        super(NT_Xent, self).__init__()
        self.temperature = temperature 
        self.batch_size = batch_size
        self.mask = self._mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def _mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def forward(self, z:list):
        batch_size = z[0].shape[0]
        N = 2 * batch_size 
        
        # z = [z_i, z_j]
        z = torch.cat(z, dim=0)
        
        # Similarity Matrix         
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        
        # Postivie / Negative pair 
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        # loss calculate 
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss = loss / N 
        loss = loss.clone()
        
        return loss 
        