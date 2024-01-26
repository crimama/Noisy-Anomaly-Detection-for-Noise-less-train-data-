import numpy as np 
import torch 
import torch.nn as nn 
import timm 
from .softpatch.src.softpatch import SoftPatch
from .softpatch.src import backbones, common 
from .softpatch.main import get_sampler 
import os 

class PatchCore(SoftPatch):
    def __init__(self, backbone:str, layers_to_extract_from:list,
                 input_shape:tuple, anomaly_score_num_nn:int,
                 sampler_name:str, sampling_ratio:float,
                 faiss_on_gpu:bool, faiss_num_workers:int,
                 lof_k:int, threshold:float,
                 weight_method:str, with_soft_weight:bool,
                 device
                ):
        super(PatchCore,self).__init__()
        
        sampler = get_sampler(sampler_name,
                              sampling_ratio,
                              device)
        
        nn_method = common.FaissNN(
            faiss_on_gpu,
            faiss_num_workers,
            int(device.strip('cuda:'))
        )
        
        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        
        self.load(
            backbone = timm.create_model(backbone, pretrained=True),
            layers_to_extract_from = layers_to_extract_from,
            device                 = device,
            input_shape            = input_shape,
            featuresampler         = sampler,
            nn_method              = nn_method,
            lof_k                  = lof_k,
            threshold              = threshold,
            weight_method          = weight_method,
            soft_weight_flag       = with_soft_weight,
            anomaly_score_num_nn   = anomaly_score_num_nn
        )        
        self.data = [] 
        
    def forward(self, images: torch.Tensor):
        if self.training:
            self.data.append(images.detach().cpu())
        return images.detach().cpu()
        
    def fit(self):
        self._fill_memory_bank(self.data)
    
    def criterion(self, outputs: torch.Tensor):
        return torch.Tensor([0.])
    
    def get_score_map(self, images: torch.Tensor) -> np.ndarray:
        '''
            score : list 
            score_map : list 
        '''
        score, score_map = self.predict(images)
        score = np.array(score)
        score_map = np.concatenate([np.expand_dims(sm,0) for sm in score_map]) #(B,W,H)
        score_map = np.expand_dims(score_map,1) # (B,1,W,H)
        return score, score_map
    
    def __init_model__(self):
        self.anomaly_scorer = common.NearestNeighbourScorer(            
            n_nearest_neighbours = self.anomaly_scorer.n_nearest_neighbours,
            nn_method            = common.FaissNN(
                                        self.faiss_on_gpu,
                                        self.faiss_num_workers,
                                        int(self.device.strip('cuda:'))
                                        )
            )
        self.data = []  