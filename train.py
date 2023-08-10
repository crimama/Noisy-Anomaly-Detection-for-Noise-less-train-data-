import logging
import wandb
import time
import os
import json

import numpy as np
import pandas as pd

import torch
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


def accuracy(outputs, targets, return_correct=False):
    # calculate accuracy
    preds = outputs.argmax(dim=1) 
    correct = targets.eq(preds).sum().item()
    
    if return_correct:
        return correct
    else:
        return correct/targets.size(0)


def calc_metrics(y_true: list, y_score: np.ndarray, y_pred: list, return_per_class: bool = False) -> dict:
    # softmax
    y_score = torch.nn.functional.softmax(torch.FloatTensor(y_score), dim=1)
    
    # metrics
    auroc = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    bcr = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        'auroc'     : auroc, 
        'f1'        : f1, 
        'recall'    : recall, 
        'precision' : precision,
        'bcr'       : bcr
    }

    if return_per_class:
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # merics per class
        f1_per_class = f1_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        acc_per_class = cm.diagonal() / cm.sum(axis=1)
    
        metrics.update({
            'per_class':{
                'cm': [NoIndent(elem) for elem in cm.tolist()],
                'f1': f1_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'precision': precision_per_class.tolist(),
                'acc': acc_per_class.tolist()
            }
        })

    return metrics


def train(model, dataloader, criterion, optimizer, accelerator: Accelerator, log_interval: int) -> dict:
    '''
    for imgs, labels in dataloader  -> imgs : [img_i, img_j]
    loss = criterion(outputs:list)
    '''   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    total_preds = []
    total_score = []
    total_targets = []
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (images, targets) in enumerate(dataloader):
        with accelerator.accumulate(model):
            data_time_m.update(time.time() - end)
            
            # predict
            outputs = [model(image) for image in images]

            # calc loss
            loss = criterion(outputs)
            accelerator.backward(loss)

            # loss update
            optimizer.step()
            optimizer.zero_grad()
            losses_m.update(loss.item())            
            
            # batch time
            batch_time_m.update(time.time() - end)

            if (idx+1) % accelerator.gradient_accumulation_steps == 0:
                if ((idx+1) // accelerator.gradient_accumulation_steps) % log_interval == 0: 
                    _logger.info(f'TRAIN [{(idx+1)//accelerator.gradient_accumulation_steps}/{len(dataloader)//accelerator.gradient_accumulation_steps}] Loss: {losses_m.val:>6.4f} ({losses_m.avg:>6.4f}) ')
    
            end = time.time()
    
    # calculate metrics
    # metrics = calc_metrics(
    #     y_true  = total_targets,
    #     y_score = total_score,
    #     y_pred  = total_preds
    # )
    
    # metrics.update([('acc',acc_m.avg), ('loss',losses_m.avg)])
    
    # # logging metrics
    # _logger.info('\nTRAIN: Loss: %.3f | Acc: %.3f%% | BCR: %.3f%% | AUROC: %.3f%% | F1-Score: %.3f%% | Recall: %.3f%% | Precision: %.3f%%\n' % 
    #              (metrics['loss'], 100.*metrics['acc'], 100.*metrics['bcr'], 100.*metrics['auroc'], 100.*metrics['f1'], 100.*metrics['recall'], 100.*metrics['precision']))
    
    # # classification report
    # _logger.info(classification_report(y_true=total_targets, y_pred=total_preds, digits=4))
    
    # return metrics
        
        
def test(model, dataloader, criterion, log_interval: int, name: str = 'TEST', return_per_class: bool = False) -> dict:
    correct = 0
    total = 0
    total_loss = 0
    
    total_preds = []
    total_score = []
    total_targets = []
    
    model.eval()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            # predict
            outputs = [model(image) for image in images]
            
            # loss 
            loss = criterion(outputs)
            
            # # total loss and acc
            # if isinstance(outputs, dict):
            #     outputs = outputs['logits']
            # total_loss += loss.item()
            # correct += accuracy(outputs, targets, return_correct=True)
            # total += targets.size(0)
            
            # # stack output
            # total_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            # total_score.extend(outputs.cpu().tolist())
            # total_targets.extend(targets.cpu().tolist())
            
            # if (idx+1) % log_interval == 0: 
            #     _logger.info('{0:s} [{1:d}/{2:d}]: Loss: {3:.3f} | Acc: {4:.3f}% [{5:d}/{6:d}]'.format(name, idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))
    
    # calculate metrics
    # metrics = calc_metrics(
    #     y_true           = total_targets,
    #     y_score          = total_score,
    #     y_pred           = total_preds,
    #     return_per_class = return_per_class
    # )
    
    # metrics.update([('acc',correct/total), ('loss',total_loss/len(dataloader))])
    
    # # logging metrics
    # _logger.info('\n%s: Loss: %.3f | Acc: %.3f%% | BCR: %.3f%% | AUROC: %.3f%% | F1-Score: %.3f%% | Recall: %.3f%% | Precision: %.3f%%\n' % 
    #              (name, metrics['loss'], 100.*metrics['acc'], 100.*metrics['bcr'], 100.*metrics['auroc'], 100.*metrics['f1'], 100.*metrics['recall'], 100.*metrics['precision']))
    
    # # classification report
    # _logger.info(classification_report(y_true=total_targets, y_pred=total_preds, digits=4))
    
    # return metrics
            
                
def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler, accelerator: Accelerator,
    epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None, ckp_metric: str = None
) -> None:

    step = 0
    best_score = 0
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
            criterion    = criterion, 
            log_interval = log_interval,
            name         = 'VALID'
        )

        # wandb
        # if use_wandb:
        #     metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        #     metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        #     metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        #     wandb.log(metrics, step=step)
        
        step += 1
        
        # update scheduler  
        if scheduler:
            scheduler.step()
        
        # checkpoint - save best results and model weights
        # if ckp_metric:
        #     ckp_cond = (best_score > eval_metrics[ckp_metric]) if ckp_metric == 'loss' else (best_score < eval_metrics[ckp_metric])
        #     if savedir and ckp_cond:
        #         best_score = eval_metrics[ckp_metric]
        #         state = {'best_step':step}
        #         state.update(eval_metrics)
        #         json.dump(state, open(os.path.join(savedir, f'results_seed{seed}_best.json'), 'w'), indent='\t')
        #         torch.save(model.state_dict(), os.path.join(savedir, f'model_seed{seed}_best.pt'))
    

def al_run(
    exp_name: str, modelname: str, pretrained: bool,
    strategy: str, n_start: int, n_end: int, n_query: int, n_subset: int, 
    init_method: str, init_method_params: dict,
    trainset, validset, testset,
    criterion_name: str, criterion_params: dict,
    img_size: int, num_classes: int, batch_size: int, test_batch_size: int, num_workers: int, 
    opt_name: str, lr: float, opt_params: dict,
    epochs: int, log_interval: int, use_wandb: bool, savedir: str, seed: int, accelerator: Accelerator, ckp_metric: str = None, cfg: dict = None):
    
    assert cfg != None if use_wandb else True, 'If you use wandb, configs should be exist.'
    
    # set active learning arguments
    nb_round = (n_start - n_end)/n_query
    
    if nb_round % int(nb_round) != 0:
        nb_round = int(nb_round) + 1
    else:
        nb_round = int(nb_round)
    
    # logging
    _logger.info('[total samples] {}, [initial samples] {} [qeury samples] {} [end samples] {} [total round] {}'.format(
        len(trainset), n_start, n_query, n_end, nb_round))
    
    # inital sampling labeling
    labeled_idx = create_labeled_index(
        method   = init_method,
        trainset = trainset,
        size     = n_start,
        seed     = seed,
        **init_method_params
    )
    
    # create model 
    model = ResNetSimCLR(
                    modelname  = cfg.MODEL.modelname,
                    img_size   = cfg.DATASET.img_size,
                    pretrained = cfg.MODEL.pretrained,
                    out_dim    = cfg.MODEL.out_dim
                    )
    
    # create criterion 
    criterion = __import__('criterions').__dict__[criterion_name](
                    batch_size = batch_size,
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
    
    # run
    #! nb_round 가 마이넉스가 나는 문제 해결 
    for r in range(nb_round+1):
        
        if r != 0:    
            # query sampling    
            query_idx = strategy.query(model, n_subset=n_subset)
            

            # clean memory
            del model, optimizer, scheduler, trainloader, validloader, testloader
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
            wandb.init(name=f'{exp_name}_round{r}', project=cfg.TRAIN.wandb.project_name, entity=cfg.TRAIN.wandb.entity, config=OmegaConf.to_container(cfg))

        # fitting model
        fit(
            model        = model, 
            trainloader  = trainloader, 
            testloader   = testloader, 
            criterion    = criterion, 
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            epochs       = epochs, 
            use_wandb    = use_wandb,
            log_interval = log_interval,
            savedir      = savedir if validset != testset else None,
            seed         = seed if validset != testset else None,
            ckp_metric   = ckp_metric if validset != testset else None
        )
        
        # save model
        # torch.save(model.state_dict(), os.path.join(savedir, f"model_seed{seed}.pt"))

        # # load best checkpoint 
        # if validset != testset:
        #     model.load_state_dict(torch.load(os.path.join(savedir, f'model_seed{seed}_best.pt')))
        
        test_results = test(
            model            = model, 
            dataloader       = testloader, 
            criterion        = strategy.loss_fn, 
            log_interval     = log_interval,
            return_per_class = True
        )
                
        wandb.finish()