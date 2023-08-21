# UAADF : Unsupervised Active Anomaly Detection Framework 
- Anomaly Detection Framework for Fully unsupervised learning 
- Using Active learning query strategy for Data Refinement 


# Environments

NVIDIA pytorch docker [ [link](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html#rel-22-12) ]

```bash
docker pull nvcr.io/nvidia/pytorch:22.12-py3
```

`requirements.txt`

```bash
accelerate==0.18.0
wandb
torchvision==0.15.0a0
gradio==3.27.0
omegaconf
timm==0.9.2
seaborn
easydict 
pyDantic
```


# Methods

## Baseline
`./models`
- STPM : https://arxiv.org/abs/2103.04257
- SimCLR : https://arxiv.org/abs/2002.05709

## Query Strategy 
`./query_strategies`

- ~~Random Sampling~~
- Entropy Sampling

## Run 
```bash
anomaly_ratio='0 0.05 0.1 0.15'
query_strategy='entropy_sampling random_sampling margin_sampling least_confidence'
class_name='leather zipper metal_nut wood pill grid tile capsule hazelnut toothbrush screw carpet bottle cable all'

for q in $query_strategy
do
    for r in $anomaly_ratio
    do
        for c in $class_name
        do
            echo "query_strategy: $q, anomaly_ratio: $r, class_name: $c"
            python main.py --default_setting configs/benchmark/default_setting.yaml \
                        --strategy_setting configs/benchmark/$q.yaml \
                        DATASET.anomaly_ratio $r \
                        DATASET.params.class_name $c
        done
    done
done


```

# Result 

[result.ipynb](result.ipynb)