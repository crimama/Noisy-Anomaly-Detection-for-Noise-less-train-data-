import numpy as np
import os
import random
import wandb
import torch
import logging
from arguments import parser

from train import refinement_run
from datasets import create_dataset
from utils.log import setup_default_logging

from accelerate import Accelerator
from omegaconf import OmegaConf

from torch.distributed import get_rank
torch.autograd.set_detect_anomaly(True)

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def run(cfg):

    # set accelerator
    accelerator = Accelerator(
        mixed_precision             = cfg.TRAIN.mixed_precision
    )

    setup_default_logging()
    torch_seed(cfg.DEFAULT.seed)

    # set device
    _logger.info('Device: {}'.format(accelerator.device))

    # load dataset
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
    
    # make save directory
    savedir = os.path.join(
                                cfg.DEFAULT.savedir,
                                cfg.DATASET.dataset_name,
                                cfg.MODEL.method,
                                cfg.DATASET.class_name,
                                cfg.DEFAULT.exp_name,
                                f"seed_{cfg.DEFAULT.seed}"
                            )
    
    # assert not os.path.isdir(savedir), f'{savedir} already exists'
    os.makedirs(savedir, exist_ok=True)
    
    # save config
    OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
    
    # run active learning
    refinement_run(
        exp_name         = cfg.DEFAULT.exp_name,

        # Model 
        method           = cfg.MODEL.method,
        backbone         = cfg.MODEL.backbone,
        model_params     = cfg.MODEL.get('params',{}),

        # Dataset
        trainset         = trainset,
        testset          = testset,
        
        # Refinement 
        nb_round         = cfg.REFINEMENT.nb_round,

        # Train
        batch_size       = cfg.DATASET.batch_size,
        test_batch_size  = cfg.DATASET.test_batch_size,
        num_workers      = cfg.DATASET.num_workers,

        opt_name         = cfg.OPTIMIZER.opt_name,
        lr               = cfg.OPTIMIZER.lr,
        opt_params       = cfg.OPTIMIZER.get('params',{}),
        
        scheduler_name   = cfg.SCHEDULER.name,
        scheduler_params = cfg.SCHEDULER.get('params',{}),

        epochs           = cfg.TRAIN.epochs,
        log_interval     = cfg.TRAIN.log_interval,
        use_wandb        = cfg.TRAIN.wandb.use,

        savedir          = savedir,
        seed             = cfg.DEFAULT.seed,
        accelerator      = accelerator,
        cfg              = cfg
    )
    

if __name__=='__main__':

    # config
    cfg = parser()
    
    # run
    run(cfg)