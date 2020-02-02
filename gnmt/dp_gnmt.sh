#! /bin/bash
set -x
DATA_SMALL=./opensource/iwslt15
DATA_LARGE=./opensource/wmt16

DATA=${DATA_LARGE}
MODEL=gnmt

###
NUM_LAYERS=8
NUM_GPUS_PER_WORKER=1
OVERWRITE_LOADED_PARAMS=False

mkdir -p nmt_attention_model

CUDA_VISIBLE_DEVICES=2 python -m nmt.nmt \
    --src=en --tgt=de \
    --hparams_path=nmt/standard_hparams/wmt16_gnmt_8_layer.json \
    --vocab_prefix=${DATA}/vocab.bpe.32000  \
    --train_prefix=${DATA}/train.tok.clean.bpe.32000 \
    --dev_prefix=${DATA}/newstest2013.tok.bpe.32000  \
    --test_prefix=${DATA}/newstest2015.tok.bpe.32000 \
    --num_layers=${NUM_LAYERS} \
    --num_gpus=${NUM_GPUS_PER_WORKER} \
    --pass_hidden_state=False \
    --out_dir=./nmt_attention_model \
    --log_device_placement=True \
    --override_loaded_hparams=${OVERWRITE_LOADED_PARAMS} \
    --fake_io=True \
    > output2.log 2>&1 &
