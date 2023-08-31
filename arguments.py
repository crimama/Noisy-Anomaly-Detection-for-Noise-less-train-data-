from omegaconf import OmegaConf
import argparse
from datasets import stats
from easydict import EasyDict


def convert_type(value):
    # None
    if value == 'None':
        return None
    
    # list or tuple
    elif len(value.split(',')) > 1:
        return value.split(',')
    
    # bool
    check, value = str_to_bool(value)
    if check:
        return value
    
    # float
    check, value = str_to_float(value)
    if check:
        return value
    
    # int
    check, value = str_to_int(value)
    if check:
        return value
    
    return value

def str_to_bool(value):
    try:
        check = isinstance(eval(value), bool)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value
    
def str_to_float(value):
    try:
        check = isinstance(eval(value), float)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value
    
def str_to_int(value):
    try:
        check = isinstance(eval(value), int)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value
    
def get_parser():
    parser = argparse.ArgumentParser(description='Active Learning - Benchmark')
    parser.add_argument('--default_setting', type=str, default=None, help='default config file')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args 

def parser(jupyter:bool = False, default_setting:str = None):
    
    if jupyter:
        args = default_setting 
    else:
        args = get_parser()
        default_setting = args.default_setting 
        
    # load default config
    cfg = OmegaConf.load(default_setting)
    
    # Update experiment name
    cfg.DEFAULT.exp_name = cfg.AL.strategy if 'AL' in cfg.keys() else 'Full'
    
    # update cfg
    if not jupyter:
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            if k == 'DEFAULT.exp_name':
                cfg.DEFAULT.exp_name = f'{cfg.DEFAULT.exp_name}-{v}'
            else:
                OmegaConf.update(cfg, k, convert_type(v), merge=True)
       
    # load dataset statistics
    if cfg.DATASET.dataset_name == 'MVTecAD':
        cfg.DATASET.update(stats.datasets[cfg.DATASET.class_name])
    else:    
        cfg.DATASET.update(stats.datasets[cfg.DATASET.dataset_name])
    
    print(OmegaConf.to_yaml(cfg))
    
    return cfg  

def jupyter_parser(default_setting:str=None, strategy_setting:str=None):
    cfg = OmegaConf.load(default_setting)    
    
    if strategy_setting:
        cfg_task = OmegaConf.load(strategy_setting)
        cfg = OmegaConf.merge(cfg, cfg_task)
         
    # load dataset statistics
    if cfg.DATASET.dataset_name == 'MVTecAD':
        cfg.DATASET.update(stats.datasets[cfg.DATASET.class_name])
    else:    
        cfg.DATASET.update(stats.datasets[cfg.DATASET.dataset_name])
    
    cfg = EasyDict(OmegaConf.to_container(cfg))
    return cfg  