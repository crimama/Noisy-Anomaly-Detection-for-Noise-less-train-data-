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
from torch.utils.data import DataLoader
from collections import OrderedDict
from accelerate import Accelerator

from sklearn.metrics import roc_auc_score, auc
from skimage import measure
from statistics import mean
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
        
def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, np.ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, np.ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    #assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for i,th in enumerate(np.arange(min_th, max_th, delta)):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        
        df.loc[i,:] = [mean(pros),fpr,th]
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc        
     

def cal_metrics(img_size:int, true_labels:np.ndarray, score_maps:np.ndarray, gts=None):
    # Image Level AUROC 
    size = score_maps.shape[0]
    image_level_score = score_maps.reshape(size,-1).max(1)
    image_auroc = roc_auc_score(true_labels, image_level_score)

    # Pixel Level AUROC 
    # preprocess score for pixel 
    pixel_score = np.transpose(score_maps,(0,2,3,1))
    pixel_score = np.array([np.expand_dims(cv2.resize(score, dsize=(img_size,img_size)), axis=2) for score in pixel_score])
    
    # preprocess gt for pixel 
    gts = np.transpose(gts, (0,2,3,1))
    gts = np.array([cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) for gt in gts])
    
    # cal pixel auroc score 
    pixel_auroc = roc_auc_score(gts.flatten(), pixel_score.flatten())
    
    # Pixel Level AUPRO 
    aupro = compute_pro(gts, pixel_score.squeeze(3))
    
    return image_auroc, pixel_auroc, aupro


def train(model, dataloader, optimizer, accelerator: Accelerator, log_interval: int) -> dict:
    '''
    for imgs, labels in dataloader  -> imgs : [img_i, img_j]
    loss = criterion(outputs:list)
    '''   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (images, _, _) in enumerate(dataloader):        
        data_time_m.update(time.time() - end)
        
        # predict
        loss = model(images)

        # loss update
        if model.__class__.__name__ != 'PatchCore':
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
        losses_m.update(loss.item())            
        
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
        
    # logging metrics
    _logger.info('TRAIN: Loss: %.3f' % (losses_m.avg))
    
    train_result = {'loss' : losses_m.avg}
    return train_result 

@torch.no_grad()
def validation(model, dataloader, log_interval: int) -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    if model.__class__.__name__ != 'PatchCore':
        model.eval()
        for idx, (images, _, _) in enumerate(dataloader):   
            data_time_m.update(time.time() - end)     
            
            # predict
            loss = model(images)
            losses_m.update(loss.item())    
            
            # batch time
            batch_time_m.update(time.time() - end)
            
            
            _logger.info('Valid [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        (idx+1), 
                        len(dataloader),
                        loss       = losses_m, 
                        batch_time = batch_time_m,
                        rate       = images[0].size(0) / batch_time_m.val,
                        rate_avg   = images[0].size(0) / batch_time_m.avg,
                        data_time  = data_time_m
                        ))
            
        valid_result = {'loss' : losses_m.avg}
    else: # In case PatchCore 
        valid_result = {'loss' : 0}
    return valid_result

@torch.no_grad()
def test(model, dataloader, img_size) -> dict:
    losses_m = AverageMeter()
    
    true_labels = [] 
    score_list = [] 
    true_gts = []
    if model.__class__.__name__ != 'PatchCore':
        model.eval()
    else:
        model.eval(next(model.parameters()).device)
    for idx, (images, labels, gts) in enumerate(dataloader):
        # predict
        if model.__class__.__name__ != 'PatchCore':
            loss, outputs = model(images, only_loss = False)            
            losses_m.update(loss.item())       
            score = model.get_score_map(outputs)
        else:
            _, score = model.get_score_map(images)
            losses_m.update(0)
        
        # Stack Scoring for image level 
        true_labels.append(labels.detach().cpu().numpy())
        score_list.append(score.detach().cpu().numpy())
        true_gts.append(gts.detach().cpu().numpy())
        
        # Stack scoring for pixel level 
            
            
    image_auroc, pixel_auroc, aupro = cal_metrics(
        img_size    = img_size,
        true_labels = np.concatenate(true_labels),
        score_maps  = np.concatenate(score_list),
        gts         = np.concatenate(true_gts)
        )
    
    # logging metrics
    _logger.info('TEST: Loss: %.3f | Image AUROC: %.3f%%| Pixel AUROC: %.3f%% | Pixel AUPRO: %.3f%%' % (losses_m.avg, image_auroc, pixel_auroc, aupro))
    
    test_result = {
            'loss' : losses_m.avg,
            'image_auroc' : image_auroc,
            'pixel_auroc' : pixel_auroc,
            'aupro'       : aupro
        }
    return test_result 


def fit(
    model, trainloader, validloader, testloader, optimizer, scheduler, accelerator: Accelerator,
    img_size, r :int, epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None
) -> None:

    step = 0
    best_score = np.inf
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        
        train_metrics = train(
            model        = model, 
            dataloader   = trainloader, 
            optimizer    = optimizer, 
            accelerator  = accelerator, 
            log_interval = log_interval
        )
        
        valid_metrics = validation(
            model        = model,
            dataloader   = validloader,
            log_interval = log_interval
        )
        
        test_metrics = test(
            img_size     = img_size,
            model        = model, 
            dataloader   = testloader,
        )
        
        epoch_time_m.update(time.time() - end)
        end = time.time()
        
        if scheduler is not None:
            scheduler.step()
        
        # logging 
        metrics = OrderedDict(round=r)
        metrics.update([('epoch', epoch)])
        metrics.update([('lr',round(optimizer.param_groups[0]['lr'],5))])
        metrics.update([('train_' + k, round(v,4)) for k, v in train_metrics.items()])
        metrics.update([('valid_' + k, round(v,4)) for k, v in valid_metrics.items()])
        metrics.update([('test_' + k, round(v,4)) for k, v in test_metrics.items()])
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
        ckp_cond = best_score > valid_metrics['loss']
        if savedir and ckp_cond:
            best_score = valid_metrics['loss']
            state = {'best_step':step,
                     'valid': valid_metrics,
                     'test': test_metrics}
            json.dump(state, open(os.path.join(savedir, f'results_seed{seed}_best.json'), 'w'), indent='\t')
            torch.save(model.state_dict(), os.path.join(savedir, f'model_seed{seed}_best.pt'))
        

        
def refinement_run(
    exp_name: str, 
    method: str, model_name: str, model_params: dict,
    trainset, validset, testset,
    nb_round: int,
    batch_size: int, test_batch_size: int, num_workers: int, 
    opt_name: str, lr: float, opt_params: dict,
    epochs: int, log_interval: int, use_wandb: bool, 
    savedir: str, seed: int, accelerator: Accelerator, cfg: dict = None):
    
    assert cfg != None if use_wandb else True, 'If you use wandb, configs should be exist.'
    
    # set active learning arguments

    
    # logging
    
    # create model 
    model = __import__('models').__dict__[method](
        model_name = model_name,
        **model_params
        )    
    
    # define train dataloader
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = batch_size,
        num_workers = num_workers
    )
    
    validloader = DataLoader(
        dataset     = validset,
        batch_size  = test_batch_size,
        shuffle     = False,
        num_workers = num_workers
    )
    
    # define test dataloader
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = test_batch_size,
        shuffle     = False,
        num_workers = num_workers
    )
    
    # run
    for r in range(nb_round):
        _logger.info(f'\nRound : [{r}/{nb_round}]')          
        
        # optimizer
        optimizer = __import__('torch.optim', fromlist='optim').__dict__[opt_name](model.parameters(), lr=lr, **opt_params)

        # scheduler
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=0.00001)
        scheduler = None 
        
        
        # # prepraring accelerator
        model, optimizer, trainloader, validloader, testloader, scheduler = accelerator.prepare(
            model, optimizer, trainloader, validloader, testloader, scheduler
        )
        
        # initialize wandb
        if use_wandb:
            wandb.init(name=f'{exp_name}_round{r}', project=cfg.TRAIN.wandb.project_name, config=OmegaConf.to_container(cfg))

        # fitting model
        fit(
            model        = model, 
            trainloader  = trainloader, 
            validloader  = validloader,
            testloader   = testloader, 
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            img_size     = cfg.DATASET.img_size,
            r            = r,
            epochs       = epochs, 
            use_wandb    = use_wandb,
            log_interval = log_interval,
            savedir      = savedir ,
            seed         = seed 
        )        
                
        wandb.finish()