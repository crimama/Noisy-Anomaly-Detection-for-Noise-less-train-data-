

# anomaly_ratio='0 0.05 0.1 0.15'
# query_strategy='entropy_sampling random_sampling margin_sampling least_confidence'
# class_name='leather zipper metal_nut wood pill grid tile capsule hazelnut toothbrush screw carpet bottle cable all'

gpu_id=$1
if [ $gpu_id == '0' ];then
# class_name='leather zipper metal_nut wood pill grid tile capsule'
class_name='transistor'
elif [ $gpu_id == '1' ];then
# class_name='hazelnut toothbrush screw carpet bottle cable transistor'
class_name='tile'
fi 

# anomaly_ratio='0.2 0.04 0.06 0.08 0.1'
anomaly_ratio='0'

for r in $anomaly_ratio
do
    for c in $class_name
    do
        echo "anomaly_ratio: $r, class_name: $c"
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py --default_setting configs/benchmark/default_setting.yaml \
                    DATASET.anomaly_ratio $r \
                    DATASET.class_name $c
    done
done