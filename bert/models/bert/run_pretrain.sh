#!/usr/bin/env bash

#python bert_create_pretrain_data.py \
#  --vocab_file=./toy_bert/vocab.txt \
#  --input_file=./bak/sample_text.txt \
#  --output_file=./bak/sample_out.tfrecord
#  --do_lower_case=True \
#  --max_seq_length=128 \
#  --max_predictions_per_seq=20 \
#  --masked_lm_prob=0.15 \
#  --random_seed=12345 \
#  --dupe_factor=5

export BERT_BASE_DIR=./bert_model_pretrain/

python bert_pretrain.py \
  --train_file=./data/pretrain/sample_out.id.txt \
  --predict_file=./data/pretrain/sample_out.id.txt \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5

