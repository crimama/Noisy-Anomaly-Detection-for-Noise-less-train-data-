#anomaly_ratio='0 0.05 0.1 0.15'
#query_strategy='entropy_sampling random_sampling margin_sampling least_confidence'

gpu_id=$1

if [ $gpu_id == '0' ]; then
  class_name='automobile'
  anomaly_ratio='0.1'
elif [ $gpu_id == '1' ]; then
  class_name='cat'
  anomaly_ratio='0.1'
else
  echo "Invalid GPU ID. Please provide a valid GPU ID (0 or 1)."
fi

for c in $class_name
do
  for r in $anomaly_ratio
  do
    echo "class_name: $c"
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --default_setting configs/benchmark/rd_cifar10.yaml \
    DATASET.class_name $c \
    DATASET.params.anomaly_ratio $r
  done
done 


