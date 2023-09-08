import json 
import pandas as pd 
import numpy as np 
import os 
import seaborn as sns 
import matplotlib.pyplot as plt 
from glob import glob 

def load_df(log_dir):
    with open(log_dir, 'r') as f:
        data = f.readlines()
    df = pd.DataFrame(list(pd.Series(data).map(eval).values))    
    return df 

def load_log_data(class_name, version, result_dir,method):

    df = pd.DataFrame()
    for i,v in enumerate(version):
        log_dir = os.path.join(result_dir,class_name,method,v,'log.txt')
        
        temp_df = load_df(log_dir)
        temp_df['hue'] = v 
        
        df = pd.concat([df,temp_df])
    
    return df 

def plot_log(data,class_name, metrics:list, figsize:tuple = (10,7)):
    
    fig, axes = plt.subplots(len(metrics),1,figsize=figsize)
    
    for ax, metric in zip(axes, metrics):
        sns.lineplot(
            x = 'epoch',
            y = metric,
            data = data,
            hue = data['hue'],
            ax = ax 
        )
        
    fig.suptitle(class_name)
    plt.show()
    
    
if __name__ == '__main__':
    version = ['anomalib_loss-anomaly_ratio_0']
    result_dir = './results/benchmark/MVTecAD/'
    metrics = ['test_pixel_auroc','test_aupro']
    method = 'STPM'

    log_dirs = glob(os.path.join(result_dir,'*',method,version[0],'results_seed0_best.json'))
    df = pd.DataFrame()

    for log_dir in log_dirs:
        temp_df = pd.DataFrame(json.load(open(log_dir)))
        temp_df['class'] = log_dir.split('/')[-4]
        df = pd.concat([df,temp_df])    
    df = df.reset_index()    
    df = df.pivot(index='class', columns='index', values='test')
    print(df.reset_index())