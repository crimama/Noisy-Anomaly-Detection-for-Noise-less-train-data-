import logging
import wandb
import time
import os
import json

import numpy as np
import pandas as pd
import cv2 

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.distributed import get_rank
from collections import OrderedDict
from accelerate import Accelerator

from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, \
                            balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score
 
from query_strategies import create_query_strategy, create_labeled_index
from models import ResNetSimCLR
from utils import NoIndent, MyEncoder
from omegaconf import OmegaConf


_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_kld_score(outputs:list):
    input = F.log_softmax(outputs[2],dim=1)
    target = F.softmax(outputs[3], dim=1)        
    score = torch.Tensor([nn.KLDivLoss(reduction='batchmean')(input[i], target[i]) for i in range(len(input))])
    return score 
    

def get_score_map(outputs) -> torch.Tensor:
    '''
    outputs = [t_outputs, s_outputs, t_logit, s_logit]
    sm.shape = (B,1,64,64)
    '''
    t_outputs, s_outputs = outputs[0], outputs[1]
    score_map = 1.
    for t, s in zip(t_outputs, s_outputs):
        t,s = F.normalize(t,dim=1),F.normalize(s,dim=1) # channel wise normalize 
        sm = torch.sum((t - s) ** 2, 1, keepdim=True) # channel wise average 
        sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False) # Intepolation : (1,w,h) -> (1,64,64)
        score_map = score_map * sm 
    return score_map 

def cal_metrics(true_labels:np.ndarray, score_maps:np.ndarray, gts=None):
    # Image Level AUROC 
    if true_labels.sum() != 0:
        size = score_maps.shape[0]
        image_level_score = score_maps.reshape(size,-1).max(1)
        image_auroc = roc_auc_score(true_labels, image_level_score)
    else:
        image_auroc = 0 

    # Pixel Level AUROC 
    if gts is not None:
        if true_labels.sum() != 0:
            # preprocess score for pixel 
            pixel_score = np.transpose(score_maps,(0,2,3,1))
            pixel_score = np.array([np.expand_dims(cv2.resize(score, dsize=(224,224)), axis=2) for score in pixel_score])
            
            # preprocess gt for pixel 
            gts = np.transpose(gts, (0,2,3,1))
            gts = np.array([cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) for gt in gts])
            
            # cal pixel auroc score 
            pixel_auroc = roc_auc_score(gts.flatten(), pixel_score.flatten())
        else:
            pixel_auroc = 0 
    else:
        pixel_auroc = 0
    
    return image_auroc, pixel_auroc


def train(model, dataloader, criterion, optimizer, accelerator: Accelerator, log_interval: int) -> dict:
    '''
    for imgs, labels in dataloader  -> imgs : [img_i, img_j]
    loss = criterion(outputs:list)
    '''   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    true_labels = [] 
    score_list = [] 
    kld_score_list = []
    true_gts = []
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (images, labels, gts) in enumerate(dataloader):
        with accelerator.accumulate(model):
            data_time_m.update(time.time() - end)
            
            # predict
            outputs = model(images)

            # calc loss
            loss = criterion(outputs)
            accelerator.backward(loss)

            # loss update
            optimizer.step()
            optimizer.zero_grad()
            losses_m.update(loss.item())            
            
            # get anomaly score 
            score = get_score_map(outputs)
            kld_score = get_kld_score(outputs)
            
            true_labels.append(labels.detach().cpu().numpy())
            score_list.append(score.detach().cpu().numpy())
            kld_score_list.append(kld_score.detach().cpu().numpy())
            true_gts.append(gts.detach().cpu().numpy())
            
            # batch time
            batch_time_m.update(time.time() - end)

            if (idx+1) % accelerator.gradient_accumulation_steps == 0:
                if ((idx+1) // accelerator.gradient_accumulation_steps) % log_interval == 0: 
                    _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                                'LR: {lr:.3e} '
                                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                (idx+1)//accelerator.gradient_accumulation_steps, 
                                len(dataloader)//accelerator.gradient_accumulation_steps, 
                                loss       = losses_m, 
                                lr         = optimizer.param_groups[0]['lr'],
                                batch_time = batch_time_m,
                                rate       = images[0].size(0) / batch_time_m.val,
                                rate_avg   = images[0].size(0) / batch_time_m.avg,
                                data_time  = data_time_m
                                ))
                    
    
            end = time.time()
            
    
    # calculate metrics
    image_auroc, pixel_auroc = cal_metrics(
        true_labels = np.concatenate(true_labels),
        score_maps  = np.concatenate(score_list),
        gts         = np.concatenate(true_gts)
        )
    
    if np.concatenate(true_labels).sum() != 0:
        kld_auroc = roc_auc_score(np.concatenate(true_labels), np.concatenate(kld_score_list))
    else:
        kld_auroc = 0 
        
    # logging metrics
    _logger.info('TRAIN: Loss: %.3f | Image AUROC: %.3f%%| Pixel AUROC: %.3f%% | KLD AUROC: %.3f%%' % (losses_m.avg, image_auroc, pixel_auroc, kld_auroc))
    
    train_result = {
            'loss' : losses_m.avg,
            'image_auroc' : image_auroc,
            'pixel_auroc' : pixel_auroc,
            'kld_auroc' : kld_auroc
        }
    return train_result 
    
    
            
        
def test(model, dataloader, criterion) -> dict:
    losses_m = AverageMeter()
    
    true_labels = [] 
    score_list = [] 
    kld_score_list = []
    true_gts = []
    
    model.eval()
    with torch.no_grad():
        for idx, (images, labels, gts) in enumerate(dataloader):
            # predict
            outputs = model(images)
            
            # loss 
            loss = criterion(outputs)
            losses_m.update(loss.item())       
            
            # get anomaly score 
            score = get_score_map(outputs)
            kld_score = get_kld_score(outputs)
            
            # Stack Scoring for image level 
            true_labels.append(labels.detach().cpu().numpy())
            score_list.append(score.detach().cpu().numpy())
            kld_score_list.append(kld_score.detach().cpu().numpy())
            true_gts.append(gts.detach().cpu().numpy())
            
            # Stack scoring for pixel level 
            
            
    image_auroc, pixel_auroc = cal_metrics(
        true_labels = np.concatenate(true_labels),
        score_maps  = np.concatenate(score_list),
        gts         = np.concatenate(true_gts)
        )
    if np.concatenate(true_labels).sum() != 0:
        kld_auroc = roc_auc_score(np.concatenate(true_labels), np.concatenate(kld_score_list))
    else:
        kld_auroc = 0 
    
    # logging metrics
    _logger.info('TEST: Loss: %.3f | Image AUROC: %.3f%%| Pixel AUROC: %.3f%% | KLD AUROC: %.3f%%' % (losses_m.avg, image_auroc, pixel_auroc, kld_auroc))
    
    test_result = {
            'loss' : losses_m.avg,
            'image_auroc' : image_auroc,
            'pixel_auroc' : pixel_auroc,
            'kld_auroc' : kld_auroc
        }
    return test_result 


def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler, accelerator: Accelerator,
    nb_round :int, epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None, ckp_metric: str = None
) -> None:

    step = 0
    best_score = 0
    epoch_time_m = AverageMeter()
    
    end = time.time() 
    
    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        
        train_metrics = train(
            model        = model, 
            dataloader   = trainloader, 
            criterion    = criterion, 
            optimizer    = optimizer, 
            accelerator  = accelerator, 
            log_interval = log_interval
        )
        
        eval_metrics = test(
            model        = model, 
            dataloader   = testloader, 
            criterion    = criterion
        )
        
        epoch_time_m.update(time.time() - end)
        end = time.time()
        
        # logging 
        metrics = OrderedDict(round=nb_round)
        metrics.update([('epoch', epoch)])
        metrics.update([('lr',round(optimizer.param_groups[0]['lr'],5))])
        metrics.update([('train_' + k, round(v,4)) for k, v in train_metrics.items()])
        metrics.update([('eval_' + k, round(v,4)) for k, v in eval_metrics.items()])
        metrics.update([('epoch time',round(epoch_time_m.val,4))])
        
        with open(os.path.join(savedir, 'log.txt'),  'a') as f:
            f.write(json.dumps(metrics) + "\n")
        if use_wandb:
            wandb.log(metrics, step=step)
        
        step += 1
        
        # update scheduler  
        if scheduler:
            scheduler.step()
        
        # checkpoint - save best results and model weights
        if ckp_metric:
            ckp_cond = (best_score > eval_metrics[ckp_metric]) if ckp_metric == 'loss' else (best_score < eval_metrics[ckp_metric])
            if savedir and ckp_cond:
                best_score = eval_metrics[ckp_metric]
                state = {'best_step':step}
                state.update(eval_metrics)
                json.dump(state, open(os.path.join(savedir, f'results_seed{seed}_best.json'), 'w'), indent='\t')
                torch.save(model.state_dict(), os.path.join(savedir, f'model_seed{seed}_best.pt'))
        

        

def al_run(
    exp_name: str, modelname: str, pretrained: bool,
    strategy: str,nb_round:int, n_query: int, n_subset: int, 
    init_method: str, init_method_params: dict,
    trainset, validset, testset,
    criterion_name: str, criterion_params: dict,
    img_size: int, num_classes: int, batch_size: int, test_batch_size: int, num_workers: int, 
    opt_name: str, lr: float, opt_params: dict,
    epochs: int, log_interval: int, use_wandb: bool, savedir: str, seed: int, accelerator: Accelerator, ckp_metric: str = None, cfg: dict = None):
    
    assert cfg != None if use_wandb else True, 'If you use wandb, configs should be exist.'
    
    # set active learning arguments

    
    # logging
    _logger.info('[total samples] {}, [initial samples] {} [qeury samples] {} [end samples] {} [total round] {}'.format(
        len(trainset), len(trainset), n_query, int(len(trainset)-nb_round*n_query), nb_round))
    
    # inital sampling labeling
    labeled_idx = create_labeled_index(
        method   = init_method,
        trainset = trainset,
        size     = len(trainset),
        seed     = seed,
        **init_method_params
    )
    
    # create model 
    model = __import__('models').__dict__[cfg.MODEL.method](
        modelname = cfg.MODEL.modelname,
        out_dim   = cfg.MODEL.out_dim,
        contrastive = True if cfg.CRITERION.name == 'MixLoss' else False 
        )
    
    # create criterion 
    criterion = __import__('criterions').__dict__[criterion_name](
                    **criterion_params
                    )
    
    # select strategy    
    strategy = create_query_strategy(
        strategy_name = strategy, 
        model         = model,
        dataset       = trainset, 
        labeled_idx   = labeled_idx, 
        n_query       = n_query, 
        batch_size    = batch_size, 
        num_workers   = num_workers,
        params        = cfg.AL.get('params', {})
    )
    
    # define train dataloader
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = batch_size,
        sampler     = SubsetRandomSampler(indices=np.where(labeled_idx==True)[0]),
        num_workers = num_workers
    )
    
    # define test dataloader
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = test_batch_size,
        shuffle     = False,
        num_workers = num_workers
    )
    
    # query log 
    query_log_df = pd.DataFrame({'idx': range(len(labeled_idx))})
    query_log_df['query_round'] = None
    query_log_df.loc[abs(labeled_idx.astype(np.int)-1).astype(bool), 'query_round'] = 'round0'
    # run
    for r in range(nb_round):
        _logger.info(f'\nRound : [{r}/{nb_round}]')
        if r != 0:    
            # query sampling    
            query_idx = strategy.query(model, n_subset=n_subset)
            
            # save query index 
            query_log_df.loc[query_idx, 'query_round'] = f'round{r}'
            query_log_df.to_csv(os.path.join(savedir, 'query_log.csv'), index=False)
            

            # clean memory
            del model, optimizer, scheduler, trainloader
            accelerator.free_memory()
            
            # update query
            trainloader = strategy.update(query_idx)        
        
        # build Model
        model = strategy.init_model()
        
        # optimizer
        optimizer = __import__('torch.optim', fromlist='optim').__dict__[opt_name](model.parameters(), lr=lr, **opt_params)

        # scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=0.00001)
        
        
        # prepraring accelerator
        model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
            model, optimizer, trainloader, testloader, scheduler
        )
        
        # initialize wandb
        if use_wandb:
            wandb.init(name=f'{exp_name}_round{r}', project=cfg.TRAIN.wandb.project_name, config=OmegaConf.to_container(cfg))

        # fitting model
        fit(
            model        = model, 
            trainloader  = trainloader, 
            testloader   = testloader, 
            criterion    = criterion, 
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            nb_round     = r,
            epochs       = epochs, 
            use_wandb    = use_wandb,
            log_interval = log_interval,
            savedir      = savedir ,
            seed         = seed ,
            ckp_metric   = ckp_metric if validset != testset else None
        )
        
        # save model
        # torch.save(model.state_dict(), os.path.join(savedir, f"model_seed{seed}.pt"))

        # # load best checkpoint 
        # if validset != testset:
        #     model.load_state_dict(torch.load(os.path.join(savedir, f'model_seed{seed}_best.pt')))
        
        # test_results = test(
        #     model            = model, 
        #     dataloader       = testloader, 
        #     criterion        = criterion, 
        #     log_interval     = log_interval,
        #     return_per_class = True
        # )
                
        wandb.finish()