#!/bin/bash
inst_id=xlnet_collective_2worker_16gpu_fusion_64_mb_chief

INIT_CKPT_DIR=/path/to/pretrained_model
save_dir=finetuned_yy_estimator

num_core_per_host=8
batch_size=1
worker_hosts="ip1:port1,ip2:port2"

HOROVOD_HIERARCHICAL_ALLREDUCE=1 HVD_DEBUG=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_keras_estimator.py  \
  --embedding_dim=1024  \
  --num_token=32000  \
  --num_layer=24  \
  --num_head=16  \
  --feed_forward_dim=4096  \
  --attention_head_dim=64  \
  --shared_biases=False  \
  --num_core_per_host=${num_core_per_host}  \
  --attention_type='bi'  \
  --train_file=../xlnet_finetune/data/tfrecord_squad2.0/spiece.model.0.slen-512.qlen-64.train.tf_record \
  --model_dir=${save_dir}  \
  --init_checkpoint=${INIT_CKPT_DIR}/xlnet_model.ckpt  \
  --memory_len=64  \
  --target_len=512  \
  --train_batch_size=${batch_size}  \
  --learning_rate=3e-5  \
  --adam_epsilon=1e-6  \
  --max_save=10  \
  --warmup_steps=1000 \
  --save_steps=10000  \
  --train_steps=40000  \
  --distribution='collective' \
  --worker_hosts=${worker_hosts} \
  --task_index=0 \
  2>&1 | tee collective_num_gpus_${num_core_per_host}_train_batch_${batch_size}_task_index_0_hvd_fusion_64_HIERARCHICAL_ALLREDUCE_2w16g.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup  python main_keras_estimator.py  \
  --embedding_dim=1024  \
  --num_token=32000  \
  --num_layer=24  \
  --num_head=16  \
  --feed_forward_dim=4096  \
  --attention_head_dim=64  \
  --shared_biases=False  \
  --num_core_per_host=${num_core_per_host}  \
  --attention_type='bi'  \
  --train_file=../xlnet_finetune/data2/tfrecord_squad2.0/spiece.model.0.slen-512.qlen-64.train.tf_record \
  --model_dir=${save_dir}  \
  --init_checkpoint=${INIT_CKPT_DIR}/xlnet_model.ckpt  \
  --memory_len=64  \
  --target_len=512  \
  --train_batch_size=${batch_size}  \
  --learning_rate=3e-5  \
  --adam_epsilon=1e-6  \
  --max_save=10  \
  --warmup_steps=1000 \
  --save_steps=10000  \
  --train_steps=40000  \
  --distribution='collective' \
  --worker_hosts=${worker_hosts} \
  --task_index=1 \
  2>&1 | tee collective_num_gpus_${num_core_per_host}_train_batch_${batch_size}_task_index_1_hvd_fusion_64_log.log

