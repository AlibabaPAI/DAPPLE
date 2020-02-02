"""Contains utilities for converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from train_flags import FLAGS
import tensorflow as tf
from tensorflow.contrib.data.python.ops import threadpool
import pdb

def get_tfrecord_files(num_workers, file_pattern):
  """Split dataset by worker.

  Args:
      num_workers: String, the name of the dataset.
      file_pattern: The file pattern to use for matching the dataset source files.

  Returns:
      A file list.

  Raises:
      ValueError: If the dataset is unknown.
  """

  if FLAGS.dataset_name == 'mock_iwslt15':
    return []
  ret = []
  all_tfrecord_files = []
  if FLAGS.dataset_dir:
    dataset_dirs = FLAGS.dataset_dir.split(',')
  else:
    raise ValueError('Need to specify dataset, mock or real.')

  for dataset_dir in dataset_dirs:
    for (dirpath, dirnames, filenames) in os.walk(dataset_dir):
      for name in filenames:
        if file_pattern and not name.endswith(file_pattern):
          continue
        all_tfrecord_files.append(os.path.join(dirpath, name))
  assert len(all_tfrecord_files) >= num_workers
  all_tfrecord_files.sort()
  for i in range(len(all_tfrecord_files)):
    if i % num_workers == FLAGS.task_index:
      ret.append(all_tfrecord_files[i])
  return ret


def _create_mock_seq2seq_iterator():
  with tf.device('/cpu:0'):
    dataset = tf.data.Dataset.from_tensor_slices((
      tf.ones(shape=[FLAGS.batch_size * 10 * 8, FLAGS.mock_seq_length], dtype=tf.int64),
      tf.ones(shape=[FLAGS.batch_size * 10 * 8, FLAGS.mock_seq_length], dtype=tf.int64),
      tf.ones(shape=[FLAGS.batch_size * 10 * 8, FLAGS.mock_seq_length], dtype=tf.int64),
      tf.random.uniform(
        shape=[FLAGS.batch_size * 10 * 8],
        minval=1,
        maxval=FLAGS.mock_seq_length + 1,
        dtype=tf.int32,
        seed=None,
        name=None
      ),
      tf.random.uniform(
        shape=[FLAGS.batch_size * 10 * 8],
        minval=1,
        maxval=FLAGS.mock_seq_length + 1,
        dtype=tf.int32,
        seed=None,
        name=None
      )))
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size).prefetch(buffer_size=FLAGS.prefetch_buffer_size)

    dataset = threadpool.override_threadpool(
      dataset,
      threadpool.PrivateThreadPool(
        FLAGS.num_preprocessing_threads, display_name='input_pipeline_thread_pool'))

    mock_iterate_op = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, mock_iterate_op.initializer)

    return mock_iterate_op


def _experimental_data_namespace():
  return tf.data.experimental if hasattr(tf.data, "experimental") else tf.contrib.data

def _create_dataset_iterator(data_sources, parse_fn, reader=None):
  with tf.device("/cpu:0"):
    experimental_data_namespace = _experimental_data_namespace()
    files = tf.data.Dataset.from_tensor_slices(data_sources)
    dataset = files.apply(experimental_data_namespace.parallel_interleave(
      tf.data.TFRecordDataset,
      cycle_length=1,
      buffer_output_elements=FLAGS.batch_size * 8,
      prefetch_input_elements=FLAGS.batch_size * 8))
    if FLAGS.datasets_use_caching:
      dataset = dataset.cache()
    dataset = dataset.apply(experimental_data_namespace.shuffle_and_repeat(
      buffer_size=FLAGS.shuffle_buffer_size, count=FLAGS.num_epochs))
    dataset = dataset.apply(experimental_data_namespace.map_and_batch(
      map_func=parse_fn, batch_size=FLAGS.batch_size, num_parallel_batches=FLAGS.num_parallel_batches))

    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

    dataset = threadpool.override_threadpool(
      dataset,
      threadpool.PrivateThreadPool(
        FLAGS.num_preprocessing_threads, display_name='input_pipeline_thread_pool'))

    ds_iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, ds_iterator.initializer)

    return ds_iterator
