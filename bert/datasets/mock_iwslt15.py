from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets.dataset_base import BaseData


class mock_iwslt15(BaseData):

  def specific_example_proto(self, example):
    with tf.device("/cpu:0"):
      return None
