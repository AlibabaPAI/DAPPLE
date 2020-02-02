#! /bin/bash
set -x

np=1 ## num of processes per node
cross_node=0
device_num=2 ## Pipeline device num
batch_size=32
batch_num=8 ## num of micro batch

LOG_DEVICE_PLACEMENT=False

remote_ip=ip1
local_ip=ip2
local_test=true
dapple_test=true

inst_id=gnmt_8layer_${np}worker_${device_num}gpu_${batch_size}batch_size_${batch_num}micro_batch_no_clip_cut2_2while_loop_correct_fake_not_grpc

### Small and Large Translation dataset
DATA_SMALL=./data/gnmt_data/iwslt15
DATA_LARGE=./data/gnmt_data/wmt16
DATA=${DATA_LARGE}
use_fake_dataset=false
if [ ${use_fake_dataset} == "true" ]; then
TRAIN_DATASET=${DATA}/fake/train.tok.clean.bpe.32000.FAKE2
else
TRAIN_DATASET=${DATA}/train.tok.clean.bpe.32000
fi

### Model Parameters
NUM_LAYERS=8
OVERWRITE_LOADED_PARAMS=False
NUM_GPUS_PER_WORKER=1
NUM_WORKERS=1
if [ ${cross_node} = "0" ]; then
CUDA_VISIBLE_DEVICES=2,3 nohup python -m nmt.nmt \
    --src=en --tgt=de \
    --hparams_path=nmt/standard_hparams/wmt16_gnmt_8_layer.json \
    --vocab_prefix=${DATA}/vocab.bpe.32000  \
    --train_prefix=${TRAIN_DATASET} \
    --dev_prefix=${DATA}/newstest2013.tok.bpe.32000  \
    --test_prefix=${DATA}/newstest2015.tok.bpe.32000 \
    --num_layers=${NUM_LAYERS} \
    --num_gpus=${NUM_GPUS_PER_WORKER} \
    --pass_hidden_state=False \
    --out_dir=./nmt_attention_model \
    --log_device_placement=True \
    --override_loaded_hparams=${OVERWRITE_LOADED_PARAMS} \
    --batch_size=${batch_size} \
    --cross_pipeline=False \
    --pipeline_device_num=${device_num} \
    --micro_batch_num=${batch_num} \
    --job_name=debug \
    --task_index=0 \
    --fake_io=True \
    --gnmt16=True \
    --dapple_test=${dapple_test} \
    --worker_hosts=${worker_hosts} \
    > ${inst_id}.log 2>&1 &
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
CUDA_VISIBLE_DEVICES=2 mpirun -np $np  -bind-to none --report-bindings  --host ${ip_list} --allow-run-as-root -x NCCL_DEBUG=INFO nohup python  -m nmt.nmt \
    --src=en --tgt=de \
    --hparams_path=nmt/standard_hparams/wmt16_gnmt_8_layer.json \
    --vocab_prefix=${DATA}/vocab.bpe.32000  \
    --train_prefix=${TRAIN_DATASET} \
    --dev_prefix=${DATA}/newstest2013.tok.bpe.32000  \
    --test_prefix=${DATA}/newstest2015.tok.bpe.32000 \
    --num_layers=${NUM_LAYERS} \
    --num_gpus=${NUM_GPUS_PER_WORKER} \
    --pass_hidden_state=False \
    --out_dir=./nmt_attention_model2 \
    --log_device_placement=${LOG_DEVICE_PLACEMENT} \
    --override_loaded_hparams=${OVERWRITE_LOADED_PARAMS} \
    --batch_size=${batch_size} \
    --cross_pipeline=True \
    --pipeline_device_num=${device_num} \
    --micro_batch_num=${batch_num} \
    --job_name=worker \
    --task_index=1 \
    --fake_io=True \
    --gnmt16=True \
    --dapple_test=${dapple_test} \
    --worker_hosts=${worker_hosts} > tmp.log 2>&1 &
fi

CUDA_VISIBLE_DEVICES=3 mpirun -np $np -bind-to none --report-bindings  -bind-to none --host ${ip_list} --allow-run-as-root -x NCCL_DEBUG=INFO python  -m nmt.nmt \
    --src=en --tgt=de \
    --hparams_path=nmt/standard_hparams/wmt16_gnmt_8_layer.json \
    --vocab_prefix=${DATA}/vocab.bpe.32000  \
    --train_prefix=${TRAIN_DATASET} \
    --dev_prefix=${DATA}/newstest2013.tok.bpe.32000  \
    --test_prefix=${DATA}/newstest2015.tok.bpe.32000 \
    --num_layers=${NUM_LAYERS} \
    --num_gpus=${NUM_GPUS_PER_WORKER} \
    --pass_hidden_state=False \
    --out_dir=./nmt_attention_model \
    --log_device_placement=${LOG_DEVICE_PLACEMENT} \
    --override_loaded_hparams=${OVERWRITE_LOADED_PARAMS} \
    --batch_size=${batch_size} \
    --cross_pipeline=True \
    --pipeline_device_num=${device_num} \
    --micro_batch_num=${batch_num} \
    --job_name=worker \
    --task_index=0 \
    --fake_io=True \
    --gnmt16=True \
    --dapple_test=${dapple_test} \
    --worker_hosts=${worker_hosts} \
    > ${inst_id}.log 2>&1 &
fi
