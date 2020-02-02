#!/usr/bin/env bash

# max_predictions_per_seq is the maximum number of masked LM predictions per sequence.
# You should set this to around max_seq_length * masked_lm_prob

python bert_create_pretrain_data.py \
  --vocab_file=./bert_model_toy/vocab.txt \
  --input_file=./data/pretrain/sample_text.txt \
  --output_file=./data/pretrain/sample_out.id.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
