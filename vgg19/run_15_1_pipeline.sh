np=1
device_num=2
batch_size=480 # max = 190, 480 for 15:1 pipeline
batch_num=12
replica=15
local_ip=<LOCAL IP ADDRESS>
remote_ip=<REMOTE IP ADRESS>
local_test=false

inst_id=vgg19_${np}w${device_num}g_bs${batch_size}_bn${batch_num}_replica${replica}

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

# Run following cmd for server=${local_ip}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -np $np --host ${ip_list} --allow-run-as-root -x NCCL_DEBUG=INFO python tf-keras-dapple.py --fake_io=True --model=vgg19 --num_batches=10000 --batch_size=${batch_size} --strategy=none --cross_pipeline=True --pipeline_device_num=${device_num} --micro_batch_num=${batch_num} --num_replica=${replica} --job_name=worker --task_index=0 --worker_hosts=${worker_hosts}  # ${inst_id}.log 2>&1 &

# Run following cmd for server=${remote_ip}
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -np $np --host ${ip_list} --allow-run-as-root -x NCCL_DEBUG=INFO nohup python tf-keras-dapple.py --fake_io=True --model=vgg19 --num_batches=10000 --batch_size=${batch_size} --strategy=none --cross_pipeline=True --pipeline_device_num=${device_num} --micro_batch_num=${batch_num} --num_replica=${replica} --job_name=worker --task_index=1 --worker_hosts=${worker_hosts} #> ${inst_id}_1.log 2>&1 &



