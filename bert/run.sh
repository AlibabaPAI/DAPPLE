#!/bin/bash
set -x
baseline_test=1
device_num=2
pipe=True
batch_num=8
cross_node=1
np=2
dataset_name=squad
num_layers=48 # Optional values: 24 or 48
if [ ${num_layers} = "48" ]; then
  model_dir=./ckpt/uncased_L-48_H-1024_A-16
  batch_size=2 # bs=2 reaches peak memory cost (14.9GB) on V100
elif [ ${num_layer} = "24" ]; then
  model_dir=./ckpt/uncased_L-24_H-1024_A-16
  batch_size=6 # bs=6 reaches peak memory cost (14.7GB) on V100
else
  echo "Unrecognized num_layer! Only 24 or 48 is supported."
  exit -1
fi
export HOROVOD_HIERARCHICAL_ALLREDUCE='1'
#export HOROVOD_LOG_LEVEL='DEBUG'

export CUDA_VISIBLE_DEVICES=2,3 #4,5,6,7
echo $TF_TASK_RES_INFO

# Baseline test on single gpu without pipeline
if [ ${baseline_test} = "1" ]; then
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size=${batch_size} --protocol=grpc --dataset_name=${dataset_name} --dataset_dir=./data/tfrecords/ --loss_name=bert_squad --optimizer=adamweightdecay --linear_warmup=False --model_dir=${model_dir} --ckpt_file_name=bert_model.ckpt --model_config_file_name=bert_config.json --do_predict=False --enable_pipeline=False > baseline.log 2>&1 &

# single process
elif [ ${cross_node} = "0" ]; then
PYTHONPATH=./ python main.py --batch_size=${batch_size} --protocol=grpc --dataset_name=${dataset_name} --dataset_dir=./data/tfrecords/ --loss_name=bert_squad --optimizer=adamweightdecay --linear_warmup=False --model_dir=${model_dir} --ckpt_file_name=bert_model.ckpt --model_config_file_name=bert_config.json --do_predict=False --enable_pipeline=$pipe --pipeline_device_num=${device_num} --pipeline_micro_batch_num=${batch_num}

# two process
else
ip_list="localhost,localhost,localhost,localhost,localhost,localhost,localhost,localhost"
worker_hosts="ip1,ip2"
if [ ${device_num} = "1" ]; then
  worker_hosts="ip1"
fi

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=./ nohup mpirun -np $np --host ${ip_list} --allow-run-as-root -x NCCL_DEBUG=INFO python main.py --batch_size=${batch_size} --protocol=grpc --dataset_name=${dataset_name} --dataset_dir=./data/tfrecords/ --loss_name=bert_squad --optimizer=adamweightdecay --linear_warmup=False --model_dir=${model_dir} --ckpt_file_name=bert_model.ckpt --model_config_file_name=bert_config.json --do_predict=False --enable_pipeline=True --pipeline_device_num=${device_num}
--pipeline_micro_batch_num=${batch_num} --job_name=worker --task_index=1 --worker_hosts=${worker_hosts} --cross_pipeline=True > worker_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 PYTHONPATH=./ mpirun -np $np --host ${ip_list} --allow-run-as-root -x NCCL_DEBUG=INFO python main.py --batch_size=${batch_size} --protocol=grpc --dataset_name=${dataset_name} --dataset_dir=./data/tfrecords/ --loss_name=bert_squad --optimizer=adamweightdecay --linear_warmup=False --model_dir=${model_dir} --ckpt_file_name=bert_model.ckpt --model_config_file_name=bert_config.json --do_predict=False --enable_pipeline=True --pipeline_device_num=${device_num}
--pipeline_micro_batch_num=${batch_num} --job_name=worker --task_index=0 --worker_hosts=${worker_hosts} --cross_pipeline=True  > worker.log 2>&1 &

fi
