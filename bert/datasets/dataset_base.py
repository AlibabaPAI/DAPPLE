"""A base class for returning dataset iterator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset_utils
from train_flags import FLAGS


class BaseData(object):

  def __init__(self, data_sources=None, preprocessing_fn=None, reader=None):
    self._data_sources = data_sources
    self._preprocessing_fn = preprocessing_fn
    self._reader = reader


  def get_dataset(self):
    """Given a dataset name and a split_name returns a Dataset.
    Returns:
        Dataset Iterator.
    Raises:
        ValueError: If dataset_name is None.
    """
    with tf.device("/cpu:0"):
      if not FLAGS.dataset_name:
        raise ValueError('Name of dataset is None.')

      if FLAGS.dataset_name == 'mock_seq2seq':
        return dataset_utils._create_mock_seq2seq_iterator()

      return dataset_utils._create_dataset_iterator(self._data_sources,
                                                    self._parse_nlp_example,
                                                    self._reader)

  def _parse_nlp_example(self, example):
    with tf.device("/cpu:0"):
      features = self.specific_example_proto(example)
      return features

  def specific_example_proto(self, example):
    """Function to specific an Example proto in input pipeline."""
    raise NotImplementedError('Must be implemented by subclass.')
