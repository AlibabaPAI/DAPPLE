np=1
cross_node=1
device_num=2
batch_size=32 # max = 190
batch_num=12
replica=2
remote_ip=<SET YOUR REMOTE IP>
local_ip=<SET YOUR LOCAL IP>
local_test=true

export CUDA_VISIBLE_DEVICES=2,3
inst_id=vgg19_${np}w${device_num}g_bs${batch_size}_bn${batch_num}

# single process
if [ ${cross_node} = "0" ]; then
python tf-keras-ds.py --fake_io=True --model=vgg19 --num_batches=10000 --batch_size=${batch_size} --strategy=none

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
CUDA_VISIBLE_DEVICES=4,5,6,7 mpirun -np $np --host ${ip_list} --allow-run-as-root -x NCCL_DEBUG=INFO nohup python tf-keras-dapple.py --fake_io=True --model=vgg19 --num_batches=10000 --batch_size=${batch_size} --strategy=none --cross_pipeline=True --pipeline_device_num=${device_num} --micro_batch_num=${batch_num} --job_name=worker --task_index=1 --worker_hosts=${worker_hosts} > ${inst_id}_2.log 2>&1 &
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np $np --host ${ip_list} --allow-run-as-root -x NCCL_DEBUG=INFO python tf-keras-dapple.py --fake_io=True --model=vgg19 --num_batches=10000 --batch_size=${batch_size} --strategy=none --cross_pipeline=True --pipeline_device_num=${device_num} --micro_batch_num=${batch_num} --num_replica=${replica} --job_name=worker --task_index=0 --worker_hosts=${worker_hosts} #> ${inst_id}.log 2>&1 &
fi

