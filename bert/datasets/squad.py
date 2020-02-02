"""Provides data for the SQUAD dataset.
"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets.dataset_base import BaseData
from train_flags import FLAGS


class squad(BaseData):

  def specific_example_proto(self, example):
    with tf.device("/cpu:0"):
      features = tf.parse_single_example(
        example,
        features={
          "unique_ids": tf.FixedLenFeature([], tf.int64),
          "input_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
          "input_mask": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
          "segment_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
          "start_positions": tf.FixedLenFeature([], tf.int64),
          "end_positions": tf.FixedLenFeature([], tf.int64)
        }
      )
      return features
