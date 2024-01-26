
import tqdm 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn
import os
os.chdir('/Volume/VAD/UAADF/')
import torch
import torch.nn as nn 
import  torch.nn.functional as F 
from arguments import parser
from torch.utils.data import DataLoader
from datasets import create_dataset
from accelerate import Accelerator
import matplotlib.pyplot as plt 
import seaborn as sns 
from utils import img_show, img_cvt

from main import torch_seed
import random 

from query_strategies.sampler import SubsetSequentialSampler
from query_strategies.refinement import Refinementer


torch_seed(0)
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

def prepare(dataset, class_name, anomaly_ratio, baseline, weight_method, threshold):
    dataset = 'pc_mvtecad' if dataset == 'mvtecad' else 'pc_mvtecloco'
    default_setting = f'./configs/benchmark/{dataset}.yaml'
    cfg = parser(jupyter=True, default_setting = default_setting)
    cfg.DATASET.class_name = class_name 
    cfg.DATASET.params.anomaly_ratio = anomaly_ratio
    cfg.DATASET.params.baseline = baseline 
    cfg.MODEL.params.weight_method = weight_method 
    cfg.MODEL.params.threshold = threshold
    
    trainset, testset = create_dataset(
        dataset_name  = cfg.DATASET.dataset_name,
        datadir       = cfg.DATASET.datadir,
        class_name    = cfg.DATASET.class_name,
        img_size      = cfg.DATASET.img_size,
        mean          = cfg.DATASET.mean,
        std           = cfg.DATASET.std,
        aug_info      = cfg.DATASET.aug_info,
        **cfg.DATASET.get('params',{})
    )

    method            = cfg.MODEL.method
    backbone          = cfg.MODEL.backbone
    model_params      = cfg.MODEL.get('params',{})

    batch_size       = cfg.DATASET.batch_size
    test_batch_size  = cfg.DATASET.test_batch_size
    num_workers      = cfg.DATASET.num_workers

    # # define train dataloader
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = batch_size,
        num_workers = num_workers,
        shuffle     = False
    )

    # define test dataloader
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = test_batch_size,
        shuffle     = False,
        num_workers = num_workers
    )

    refinement = Refinementer(
            model          = __import__('models').__dict__[method](
                            backbone = backbone,
                            **model_params
                            ),
            n_query        = cfg.REFINEMENT.n_query,
            dataset        = trainset,
            unrefined_idx  = np.ones(len(trainset)).astype(np.bool8),
            batch_size     = batch_size,
            test_transform = testset.transform,
            num_workers    = num_workers
        )
    model = refinement.init_model()
    device = cfg.MODEL.params.device
    
    output = {}
    output['trainloader'], output['testloader'], output['model'], output['device']  = trainloader, testloader, model, device
    
    return output 

def train(inputs):
    trainloader, device, model = inputs['trainloader'], inputs['device'], inputs['model']
    for imgs, labels, gts in trainloader:
        output = model(imgs.to(device))
        loss = model.criterion(output)
    model.fit()
    
def evaluation(inputs, loco = False):
    testloader,  model = inputs['testloader'], inputs['model']
    from utils.metrics import MetricCalculator, loco_auroc

    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision','confusion_matrix'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision','confusion_matrix','aupro'])

    results = {} 
    for idx, (images, labels, gts) in enumerate(testloader):
        
        # predict
        if model.__class__.__name__ in ['PatchCore']:
            score, score_map = model.get_score_map(images)
                
        # Stack Scoring for metrics 
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(score, labels.type(torch.int))
        
    p_results = pix_level.compute()
    i_results = img_level.compute()
    
    
    if loco:
        results['loco_auroc'] = loco_auroc(pix_level,testloader)
        results['loco_auroc'] = loco_auroc(img_level,testloader)    
        return p_results, i_results, results 
    else:         
        return p_results, i_results

def patch_scoring(testloader, model):
    self = model 
    score_list = [] 
    for imgs, labels, gts in testloader: 
        images = imgs.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            image_scores, _, indices = self.anomaly_scorer.predict([features])
        
        score_list.append(image_scores)
    score_list = np.concatenate(score_list)
    return score_list 

def test_scoring(inputs):
    'test 데이터들의 각 anomaly score 산출'
    score_list = [] 
    score_map_list = [] 
    with torch.no_grad():
        for imgs, labels, gts in inputs['testloader']:
            score, score_map = inputs['model'].get_score_map(imgs)
            score_list.append(score)
            score_map_list.append(score_map)
    S = np.concatenate(score_list)
    SM = np.concatenate(score_map_list)
    return S, SM 

def scaling(inputs):
    inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))
    return inputs 
        
        
def get_indicies(inputs, lof:bool = False):
    '''
    denoising 한 index와 coreset index 구하는 메소드 
    '''
    self = inputs['model']
    
    train_embeddings = np.vstack([inputs['model'].embed(d.to('cuda')) for d,_,_ in inputs['trainloader']])
    features = train_embeddings
    
    if lof:
        with torch.no_grad():
            # pdb.set_trace()
            self.feature_shape = [28,28]
            patch_weight = self._compute_patch_weight(features) # <- get outlier score 

            # normalization
            # patch_weight = (patch_weight - patch_weight.quantile(0.5, dim=1, keepdim=True)).reshape(-1) + 1

            patch_weight = patch_weight.reshape(-1)
            threshold = torch.quantile(patch_weight, 1 - self.threshold)
            sampling_weight = torch.where(patch_weight > threshold, 0, 1) #! sampling_weight = denoising 한 index 
            #self.featuresampler.set_sampling_weight(sampling_weight) # <- subsampling data which has outlier score under thresholding
            #self.patch_weight = patch_weight.clamp(min=0)
    
    sample_features, sample_indices = self.featuresampler.run(features) #! sample_indices = coreset index         
                
    if lof:
        return {'denoising':sampling_weight.detach().cpu(), 'coreset': sample_indices}
    else:
        return {'coreset': sample_indices}
                
# 1024 512 256 128 
class DAE(nn.Module):
    def __init__(self, in_channels:int, noise_factor:float):
        super(DAE, self).__init__()
        self.in_c = in_channels 
        self.encoder = nn.Sequential(*[self.get_linaer_layer(int(self.in_c/(2**i)),int(self.in_c/(2**(i+1)))) for i in range(3)])
        self.decoder = nn.Sequential(*[self.get_linaer_layer(int(self.in_c/(2**i)),int(self.in_c/(2**(i-1)))) for i in range(3,0,-1)])
        
        self.noise_factor = noise_factor
        
    def get_linaer_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        if self.training:
            x = x + self.noise_factor * torch.randn(*x.size()).to(x.device)
        
        x = self.encoder(x)
        x = self.decoder(x)
        return x     
    
def get_patch_embed(model, trainloader):
    for imgs, labels, gts in trainloader:
        outputs = model(imgs)
    _ = model.forward_modules.eval()
    
    features = [] 
    with tqdm.tqdm(model.data, leave=True) as data_iterator:
        for image in data_iterator:
            with torch.no_grad():
                input_image = image.to(torch.float).to(model.device)
            feature = model._embed(input_image)
            features.append(feature)
    features = np.concatenate(features)        
    return features    

from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

def dae_training(EPOCH, features, criterion, model, scheduler, optimizer,BATCH_SIZE, device):
    dataset = CustomDataset(features)
    trainloader = DataLoader(dataset,BATCH_SIZE,shuffle=True)
    
    for e in tqdm.tqdm(range(EPOCH)):
        length = len(features)//256
        iterator = np.arange(length)
        np.random.shuffle(iterator)
        
        losses = [] 
        for data in trainloader:
            data = data.to(device)
            
            # predict
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            
            # loss update 
            optimizer.step()
            optimizer.zero_grad()
            
        
        scheduler.step()
        #print(np.mean(losses))
        
    recon_x = [] 
    testloader = DataLoader(dataset,BATCH_SIZE,shuffle=False)
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            
            # predict
            output = model(data)
            
            recon_x.append(output.detach().cpu().numpy())
    recon_x = np.concatenate(recon_x)
    return recon_x 

'''
test 이미지도 dae inference 한 것 적용하기 위한 코드 
'''
from utils.metrics import MetricCalculator
def dae_direct_evaluation(model, dae, testloader,device):
    model.eval()
    dae.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision','confusion_matrix'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision','confusion_matrix','aupro'])

    self = model 
    results = {} 
    for idx, (images, labels, gts) in enumerate(testloader):
        images = images.to(torch.float).to(device)
        _ = self.forward_modules.eval()
        
        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)
            ###
            features = dae.decoder(dae.encoder(torch.Tensor(features).to('cuda'))).detach().cpu().numpy()
            ###
        image_scores, _, indices = self.anomaly_scorer.predict([features]) 
        
        patch_scores = image_scores

        image_scores = self.patch_maker.unpatch_scores(
            image_scores, batchsize=batchsize
        )
        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
        image_scores = self.patch_maker.score(image_scores)

        patch_scores = self.patch_maker.unpatch_scores(
            patch_scores, batchsize=batchsize
        ) # Unfold : (B)
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

        masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores) # interpolation : (B,pw,ph) -> (B,W,H)

        score = [score for score in image_scores]
        score_map = [mask for mask in masks]       
        
        score = np.array(score)
        score_map = np.concatenate([np.expand_dims(sm,0) for sm in score_map]) #(B,W,H)
        score_map = np.expand_dims(score_map,1) # (B,1,W,H)
        
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(score, labels.type(torch.int))
        
    p_results = pix_level.compute()
    i_results = img_level.compute()
                
    return p_results['auroc'], i_results['auroc']



if __name__ == '__main__':
    DATASET = 'mvtecad'
    ANOMALYRATIO = 0

    EPOCH = 30
    LR = 1e-3
    BATCH_SIZE = 256

    RESULTS = [] 
    for CLASSNAME in ['bottle','carpet','grid','leather','metal_nut','pill','screw','tile','toothbrush','wood','zipper']:
        for ANOMALYRATIO in [0.0,0.05, 0.1, 0.15, 0.2]:
            inputs = prepare('mvtecad',CLASSNAME,ANOMALYRATIO,False,'lof',0.15)
            model = inputs['model']
            trainloader = inputs['trainloader']
            testloader = inputs['testloader']
            features = get_patch_embed(model, trainloader)

            dae = DAE(1024,0.01).to(model.device)


            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(dae.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCH, eta_min = 0.00001)

            recon_x = dae_training(EPOCH, features, criterion, dae, scheduler, optimizer, BATCH_SIZE, 'cuda')

            #sampling weight 
            score = np.mean((features - recon_x)**2,axis=1)**0.5
            sampling_weight = np.where(score > np.percentile(score,0.85),0,1)
            model.featuresampler.set_sampling_weight(torch.Tensor(sampling_weight).to(model.device))

            sample_features, sample_indices = model.featuresampler.run(features) # greedy search
            model.anomaly_scorer.fit(detection_features=[sample_features])
            p_auroc, i_auroc = dae_direct_evaluation(model, dae, testloader, model.device)
            result = [CLASSNAME, ANOMALYRATIO, p_auroc, i_auroc]
            RESULTS.append(result)
            
            with open('./results/MVTecAD/dae/log.txt', 'a') as f: 
                f.write(str(result)+'\n')
            