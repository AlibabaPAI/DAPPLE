from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets.dataset_base import BaseData
from train_flags import FLAGS

class iwslt15(BaseData):

  def specific_example_proto(self, example):
    with tf.device("/cpu:0"):
      features = tf.parse_single_example(
        example,
        features={
          "src_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
          "src_tokens": tf.FixedLenFeature([FLAGS.max_seq_length], tf.string),
          "src_len": tf.FixedLenFeature([], tf.int64),
          "tgt_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
          "tgt_tokens": tf.FixedLenFeature([FLAGS.max_seq_length], tf.string),
          "tgt_len": tf.FixedLenFeature([], tf.int64),
          "tgt_ids_out": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64)
        }
      )
      features["src_len"] = tf.cast(features["src_len"], tf.int32)
      features["tgt_len"] = tf.cast(features["tgt_len"], tf.int32)
      return features
