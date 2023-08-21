#accelerate launch main.py --default_setting configs/benchmark/default_setting.yaml --strategy_setting configs/benchmark/entropy_sampling.yaml
# python main.py --default_setting configs/benchmark/default_setting.yaml --strategy_setting configs/benchmark/random_sampling.yaml


# # dataname='CIFAR10LT CIFAR100LT'
# # IF='1 10 50 100 200'
# # losses='CrossEntropyLoss BalancedSoftmax'

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
            python main.py --default_setting configs/benchmark/default_setting.yaml --strategy_setting configs/benchmark/$q.yaml \
                        DATASET.anomaly_ratio $r \
                        DATASET.params.class_name $c
        done
    done
done
