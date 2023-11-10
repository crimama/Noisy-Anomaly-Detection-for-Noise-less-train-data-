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
import pickle 

from omegaconf import OmegaConf

from ignite.contrib import metrics 


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
             

def metric_logging(savedir, use_wandb, 
                    r, epoch, step,epoch_time_m,
                    optimizer, train_metrics, test_metrics):
    
    metrics = OrderedDict(round=r)
    metrics.update([('epoch', epoch)])
    metrics.update([('lr',round(optimizer.param_groups[0]['lr'],5))])
    metrics.update([('train_' + k, round(v,4)) for k, v in train_metrics.items()])
    metrics.update([('test_' + k, round(v,4)) for k, v in test_metrics.items()])
    metrics.update([('epoch time',round(epoch_time_m.val,4))])
    
    with open(os.path.join(savedir, 'log.txt'),  'a') as f:
        f.write(json.dumps(metrics) + "\n")
    if use_wandb:
        wandb.log(metrics, step=step)
    

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

    for idx, (images, _, _) in enumerate(dataloader):        
        data_time_m.update(time.time() - end)
        # predict
        output = model(images)
        loss   = model.criterion(output)

        # loss update
        if model.__class__.__name__ not in ['PatchCore']:
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
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

def test(model, dataloader) -> dict:    
    
    if model.__class__.__name__ == 'PatchCore':
        model.eval(next(model.parameters()).device) #Patchcore의 경우 embedding들에게 device를 할당해주어야 함 
    else:
        model.eval()
        
    pixel_auroc = metrics.ROC_AUC()
    image_auroc = metrics.ROC_AUC()
        
    for idx, (images, labels, gts) in enumerate(dataloader):
        
        # predict
        with torch.no_grad():
            if model.__class__.__name__ == 'PatchCore':
                score, score_map = model.get_score_map(images)
            else:
                outputs = model(images)   
                score_map = model.get_score_map(outputs).detach().cpu()
                score = score_map.reshape(score_map.shape[0],-1).max(-1)[0]
        
        # Stack Scoring for auroc 
        pixel_auroc.update((score_map.flatten(), gts.flatten()))
        image_auroc.update((score, labels))
        
        # Calculate results of evaluation 
        
    if dataloader.dataset.name != 'CIFAR10':
        p_auroc = pixel_auroc.compute()
    i_auroc = image_auroc.compute()
            
    # logging metrics
    if dataloader.dataset.name != 'CIFAR10':
        _logger.info('Image AUROC: %.3f%%| Pixel AUROC: %.3f%%' % (i_auroc,p_auroc))
    else:
        _logger.info('Image AUROC: %.3f%%' % (i_auroc))
        
    test_result = OrderedDict(image_auroc = i_auroc)
    if dataloader.dataset.name != 'CIFAR10':
        test_result.update([('pixel_auroc', round(p_auroc,4))])
    
    return test_result 


def fit(
    model, trainloader, testloader, optimizer, scheduler, accelerator: Accelerator,
    n_round :int, epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None
    ):

    best_score = 0.0
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    for step,  epoch in enumerate(range(epochs)):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        
        train_metrics = train(
            model        = model, 
            dataloader   = trainloader, 
            optimizer    = optimizer, 
            accelerator  = accelerator, 
            log_interval = log_interval
        )        
        
        test_metrics = test(
            model        = model, 
            dataloader   = testloader
        )
        
        epoch_time_m.update(time.time() - end)
        end = time.time()
        
        if scheduler is not None:
            scheduler.step()
        
        # logging 
        metric_logging(
            savedir = savedir, use_wandb = use_wandb, r = n_round, epoch = epoch, step = step,
            optimizer = optimizer, epoch_time_m = epoch_time_m,
            train_metrics = train_metrics, test_metrics = test_metrics)
        
        # if epoch%49 == 0:
        #     torch.save(model.state_dict(), os.path.join(savedir, f'model_{epoch}.pt')) 
                
        # checkpoint - save best results and model weights        
        if best_score < test_metrics['image_auroc']:
            best_score = test_metrics['image_auroc']
            print(f" New best score : {best_score} | best epoch : {epoch}")
            torch.save(model.state_dict(), os.path.join(savedir, f'model_best.pt')) 
                    
            
    
    return model 
        
def refinement_run(
    exp_name: str, 
    method: str, backbone: str, model_params: dict,
    trainset, testset,
    nb_round: int,
    batch_size: int, test_batch_size: int, num_workers: int, 
    opt_name: str, lr: float, opt_params: dict, 
    scheduler_name: str, scheduler_params: dict, 
    epochs: int, log_interval: int, use_wandb: bool, 
    savedir: str, seed: int, accelerator: Accelerator, cfg: dict = None):
    
    assert cfg != None if use_wandb else True, 'If you use wandb, configs should be exist.'
        
    # logging
    
    # create model 
    model = __import__('models').__dict__[method](
        backbone = backbone,
        **model_params
        )    
    
    # define train dataloader
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = batch_size,
        num_workers = num_workers,
        shuffle     = True 
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
        if scheduler_name is not None:
            scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[scheduler_name](optimizer, **scheduler_params)
        else:
            scheduler = None
        
        
        # # prepraring accelerator
        model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
            model, optimizer, trainloader, testloader, scheduler
        )
        
        # initialize wandb
        if use_wandb:
            wandb.init(name=f'{exp_name}_round{r}', project=cfg.TRAIN.wandb.project_name, config=OmegaConf.to_container(cfg))

        # fitting model
        last_model = fit(
            model        = model, 
            trainloader  = trainloader, 
            testloader   = testloader, 
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            n_round      = r,
            epochs       = epochs, 
            use_wandb    = use_wandb,
            log_interval = log_interval,
            savedir      = savedir ,
            seed         = seed 
        )        
        
        wandb.finish()
        
    torch.save(last_model.state_dict(),os.path.join(savedir, f'model_last.pt'))
    
                