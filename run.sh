

# anomaly_ratio='0 0.05 0.1 0.15'
# query_strategy='entropy_sampling random_sampling margin_sampling least_confidence'
# class_name='leather zipper metal_nut wood pill grid tile capsule hazelnut toothbrush screw carpet bottle cable all'

gpu_id=$1
if [ $gpu_id == '0' ];then
class_name='leather zipper metal_nut wood pill grid tile capsule'
elif [ $gpu_id == '1' ];then
class_name='hazelnut toothbrush screw carpet bottle cable transistor'
fi 


# anomaly_ratio='0.02 0.04 0.06 0.08 0.1'
normal_ratio='0.25'
anomaly_ratio='0'

for nr in $normal_ratio
do
    for r in $anomaly_ratio
    do
        for c in $class_name
        do
            echo "anomaly_ratio: $r, normal_ratio: $nr class_name: $c"
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --default_setting configs/benchmark/patchcore.yaml \
                        DATASET.anomaly_ratio $r \
                        DATASET.class_name $c \
                        DATASET.params.normal_ratio $nr
        done
    done
done