# Active Learning 

UAADF : Unsupervised Active Anomaly Detection Framework 

# Environments

NVIDIA pytorch docker [ [link](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html#rel-22-12) ]

```bash
docker pull nvcr.io/nvidia/pytorch:22.12-py3
```

`requirements.txt`

```bash
accelerate
wandb
torchvision
```


# Methods

`./query_strategies`

- Random Sampling (baseline)
- Entropy Sampling



