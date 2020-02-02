np=1
cross_node=1
device_num=2
batch_size=8
batch_num=8

num_cells=6
reduction_size=512

remote_ip=<SET YOUR REMOTE IP>
local_ip=<SET YOUR LOCAL IP>
local_test=true

export CUDA_VISIBLE_DEVICES=6,7
inst_id=amoeba_${np}w${device_num}g_cells${num_cells}_rdc${reduction_size}_bs${batch_size}_bn${batch_num}

# bs=32, cells=6, size=512  8 + 6 GB, 624ms
# bs=16, cells=6, size=512  4 + 6 GB, 364ms
# bs=8, cells=6, size=512  2 + 6 GB, 240ms
# cells=72, size=512  GPipe


if [ ${cross_node} = "0" ]; then
nohup python amoeba_net.py \
  --use_tpu=False \
  --data_dir=./data/ \
  --num_cells=${num_cells} \
  --reduction_size=${reduction_size} \
  --image_size=224 \
  --max_steps=2000 \
  --num_epochs=35 \
  --train_batch_size=${batch_size} \
  --eval_batch_size=${batch_size} \
  --lr=2.56 \
  --lr_decay_value=0.88 \
  --lr_warmup_epochs=0.35 \
  --moving_average_decay=0 \
  --enable_hostcall=False \
  --mode=train > ${inst_id}.log 2>&1 &

else
ip_list="localhost,localhost,localhost,localhost,localhost,localhost,localhost,localhost"
worker_hosts=${local_ip}
if [ ${device_num} != "1" ]; then
  if [ ${local_test} != "true" ]; then
    worker_hosts=${worker_hosts},${remote_ip}
  else
    worker_hosts=${local_ip},${local_ip}
  fi
fi
echo ${worker_hosts}

if [ ${local_test} == "true" ]; then
CUDA_VISIBLE_DEVICES=4,5,6,7 mpirun -np $np --host ${ip_list} --allow-run-as-root -x NCCL_DEBUG=INFO nohup python amoeba-dapple.py \
  --use_tpu=False \
  --data_dir=./data/ \
  --num_cells=${num_cells} \
  --reduction_size=${reduction_size} \
  --image_size=224 \
  --max_steps=2000 \
  --num_epochs=35 \
  --train_batch_size=${batch_size} \
  --eval_batch_size=${batch_size} \
  --lr=2.56 \
  --lr_decay_value=0.88 \
  --lr_warmup_epochs=0.35 \
  --moving_average_decay=0 \
  --enable_hostcall=False \
  --mode=train \
  --cross_pipeline=True \
  --pipeline_device_num=${device_num} \
  --micro_batch_num=${batch_num} \
  --job_name=worker \
  --task_index=1 \
  --worker_hosts=${worker_hosts} > ${inst_id}_2.log 2>&1 &
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np $np --host ${ip_list} --allow-run-as-root -x NCCL_DEBUG=INFO python amoeba-dapple.py \
  --use_tpu=False \
  --data_dir=./data/ \
  --num_cells=${num_cells} \
  --reduction_size=${reduction_size} \
  --image_size=224 \
  --max_steps=2000 \
  --num_epochs=35 \
  --train_batch_size=${batch_size} \
  --eval_batch_size=${batch_size} \
  --lr=2.56 \
  --lr_decay_value=0.88 \
  --lr_warmup_epochs=0.35 \
  --moving_average_decay=0 \
  --enable_hostcall=False \
  --mode=train \
  --cross_pipeline=True \
  --pipeline_device_num=${device_num} \
  --micro_batch_num=${batch_num} \
  --job_name=worker \
  --task_index=0 \
  --worker_hosts=${worker_hosts} #> ${inst_id}.log 2>&1 &
fi
