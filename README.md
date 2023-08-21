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

# Feedback 
- **평가 데이터셋**
    - anomaly ratio에만 집중한 나머지 test 데이터셋을 고정하지 못 했음 → anomaly ratio가 바뀜에 따라 test set 도 바뀌어서 공정한 비교가 되지 못 함
- **불완전한 학습**
    - 30epoch으로 학습을 진행 했는데 학습이 완전히 수렴되지 않은 상태로 끝난 것으로 보임
- **베이스라인의 온전한 구축**
    - 기존 세팅 대로 온전히 학습 된 것을 확인하고 이후 실험을 진행 했어야 했는데 그러지 못 함
    - 기본 성능 자체가 너무 낮게 나옴
    - 그리고 베이스라인에서 일부 모듈을 수정했기 때문에 성능 또한 온전히 재현 될 것이라는 보장이 안됌
    - simclr의 한계 → 높은 batch size 요구
        - but 사용한 데이터 수가 적음
- **일부 class 누락**
    - bash 파일에서 잘못해서 일부 class 누락 됨