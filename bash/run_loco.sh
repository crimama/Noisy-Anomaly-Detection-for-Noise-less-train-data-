#anomaly_ratio='0 0.05 0.1 0.15'
#query_strategy='entropy_sampling random_sampling margin_sampling least_confidence'
# 'capsule cable bottle carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'
gpu_id=$1

if [ $gpu_id == '0' ]; then
  class_name='capsule cable bottle carpet grid hazelnut leather'
  anomaly_ratio='0.0'
elif [ $gpu_id == '1' ]; then
  class_name='breakfast_box juice_bottle pushpins screw_bag splicing_connectors'
  anomaly_ratio='0.0 0.05 0.1 0.15 0.2'
  sampling_method='identity nearest gaussian lof'  
else
  echo "Invalid GPU ID. Please provide a valid GPU ID (0 or 1)."
fi
  
for s in $sampling_method
  do
  for c in $class_name
  do
    for r in $anomaly_ratio
    do
      echo "class_name: $c"      
      CUDA_VISIBLE_DEVICES=$gpu_id python main.py --default_setting configs/benchmark/pc_mvtecloco.yaml \
      DATASET.class_name $c \
      DATASET.params.anomaly_ratio $r \
      MODEL.params.weight_method $s \
      DEFAULT.exp_name "$s"
      done
  done
done