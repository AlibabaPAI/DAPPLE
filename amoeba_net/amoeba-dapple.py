# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=line-too-long
r"""TensorFlow AmoebaNet Example.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import math
import json
import os
from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import config_pb2
import amoeba_net_model as model_lib
import amoeba_net_slice_model as slice_model_lib
import horovod.tensorflow as hvd
from common import cluster_utils
from common import inference_warmup
import pdb

# Cloud TPU Cluster Resolvers

flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# General Parameters
flags.DEFINE_integer(
    'num_shards', 8,
    'Number of shards (TPU cores).')

flags.DEFINE_integer(
    'distributed_group_size', 1,
    help='Size of the distributed batch norm. group.'
    'Default is normalization over local examples only.'
    'When set to a value greater than 1, it will enable'
    'a distribtued batch norm. To enable a global batch norm.'
    'set distributed_group_size to FLAGS.num_shards')

flags.DEFINE_bool(
    'use_tpu', True,
    'Use TPUs rather than CPU or GPU.')

flags.DEFINE_string(
    'data_dir', '',
    'Directory where input data is stored')

flags.DEFINE_string(
    'model_dir', None,
    'Directory where model output is stored')

flags.DEFINE_string(
    'export_dir', None,
    'The directory where the exported SavedModel will be stored.')

flags.DEFINE_integer(
    'iterations_per_loop', None,
    'Number of iterations per TPU training loop.')

flags.DEFINE_integer(
    'max_steps', 1000,
    'max steps for training')

flags.DEFINE_integer(
    'train_batch_size', 256,
    'Global (not per-shard) batch size for training')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Global (not per-shard) batch size for evaluation')

flags.DEFINE_float(
    'num_epochs', 48.,
    'Number of steps use for training.')

flags.DEFINE_float(
    'num_epochs_per_eval', 1.,
    'Number of training epochs to run between evaluations.')

flags.DEFINE_string(
    'mode', 'train_and_eval',
    'Mode to run: train, eval, train_and_eval, or predict')

flags.DEFINE_integer(
    'save_checkpoints_steps', None,
    'Interval (in steps) at which the model data '
    'should be checkpointed. Set to 0 to disable.')

flags.DEFINE_bool(
    'enable_hostcall', True,
    'Skip the host_call which is executed every training step. This is'
    ' generally used for generating training summaries (train loss,'
    ' learning rate, etc...). When --enable_hostcall=True, there could'
    ' be a performance drop if host_call function is slow and cannot'
    ' keep up with the TPU-side computation.')

# Model specific parameters
flags.DEFINE_bool('use_aux_head', True, 'Include aux head or not.')
flags.DEFINE_float(
    'aux_scaling', 0.4, 'Scaling factor of aux_head')
flags.DEFINE_float(
    'batch_norm_decay', 0.9, 'Batch norm decay.')
flags.DEFINE_float(
    'batch_norm_epsilon', 1e-5, 'Batch norm epsilon.')
flags.DEFINE_float(
    'dense_dropout_keep_prob', None, 'Dense dropout keep probability.')
flags.DEFINE_float(
    'drop_connect_keep_prob', 1.0, 'Drop connect keep probability.')
flags.DEFINE_string(
    'drop_connect_version', None, 'Drop connect version.')
flags.DEFINE_string(
    'cell_name', 'amoeba_net_d', 'Which network to run.')
flags.DEFINE_integer(
    'num_cells', 12, 'Total number of cells.')
flags.DEFINE_integer(
    'reduction_size', 256, 'Default cell reduction size.')
flags.DEFINE_integer(
    'stem_reduction_size', 32, 'Stem filter size.')
flags.DEFINE_float(
    'weight_decay', 4e-05, 'Weight decay for slim model.')
flags.DEFINE_integer(
    'num_label_classes', 1001, 'The number of classes that images fit into.')

# Training hyper-parameters
flags.DEFINE_float(
    'lr', 0.64, 'Learning rate.')
flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'Optimizer (one of sgd, rmsprop, momentum)')
flags.DEFINE_float(
    'moving_average_decay', 0.9999,
    'moving average decay rate')
flags.DEFINE_float(
    'lr_decay_value', 0.9,
    'Exponential decay rate used in learning rate adjustment')
flags.DEFINE_integer(
    'lr_num_epochs_per_decay', 1,
    'Exponential decay epochs used in learning rate adjustment')
flags.DEFINE_string(
    'lr_decay_method', 'exponential',
    'Method of decay: exponential, cosine, constant, stepwise')
flags.DEFINE_float(
    'lr_warmup_epochs', 3.0,
    'Learning rate increased from zero linearly to lr for the first '
    'lr_warmup_epochs.')
flags.DEFINE_float('gradient_clipping_by_global_norm', 0,
                   'gradient_clipping_by_global_norm')

flags.DEFINE_integer(
    'image_size', 299, 'Size of image, assuming image height and width.')

flags.DEFINE_integer(
    'num_train_images', 1281167, 'The number of images in the training set.')
flags.DEFINE_integer(
    'num_eval_images', 50000, 'The number of images in the evaluation set.')

flags.DEFINE_bool(
    'use_bp16', True, 'If True, use bfloat16 for activations')

flags.DEFINE_integer(
    'eval_timeout', 60*60*24,
    'Maximum seconds between checkpoints before evaluation terminates.')

# Inference configuration.
flags.DEFINE_bool(
    'add_warmup_requests', False,
    'Whether to add warmup requests into the export saved model dir,'
    'especially for TPU inference.')
flags.DEFINE_string('model_name', 'amoeba_net',
                    'Serving model name used for the model server.')
flags.DEFINE_multi_integer(
    'inference_batch_sizes', [8],
    'Known inference batch sizes used to warm up for each core.')
# dapple config
flags.DEFINE_string('worker_hosts', '', 'worker_hosts')
flags.DEFINE_string('job_name', '', 'worker or ps')
flags.DEFINE_integer('task_index', 0, '')
flags.DEFINE_bool('cross_pipeline', False, '')
flags.DEFINE_integer('pipeline_device_num', 1, '')
flags.DEFINE_integer('micro_batch_num', 1, '')
flags.DEFINE_integer('num_replica', 1, '')

FLAGS = flags.FLAGS


# TODO(ereal): simplify this.
def override_with_flags(hparams):
  """Overrides parameters with flag values."""
  override_flag_names = [
      'aux_scaling',
      'train_batch_size',
      'batch_norm_decay',
      'batch_norm_epsilon',
      'dense_dropout_keep_prob',
      'drop_connect_keep_prob',
      'drop_connect_version',
      'eval_batch_size',
      'gradient_clipping_by_global_norm',
      'lr',
      'lr_decay_method',
      'lr_decay_value',
      'lr_num_epochs_per_decay',
      'moving_average_decay',
      'image_size',
      'num_cells',
      'reduction_size',
      'stem_reduction_size',
      'num_epochs',
      'num_epochs_per_eval',
      'optimizer',
      'enable_hostcall',
      'use_aux_head',
      'use_bp16',
      'use_tpu',
      'lr_warmup_epochs',
      'weight_decay',
      'num_shards',
      'distributed_group_size',
      'num_train_images',
      'num_eval_images',
      'num_label_classes',
  ]
  for flag_name in override_flag_names:
    flag_value = getattr(FLAGS, flag_name, 'INVALID')
    if flag_value == 'INVALID':
      tf.logging.fatal('Unknown flag %s.' % str(flag_name))
    if flag_value is not None:
      _set_or_add_hparam(hparams, flag_name, flag_value)


def build_hparams():
  """Build tf.Hparams for training Amoeba Net."""
  hparams = model_lib.build_hparams(FLAGS.cell_name)
  override_with_flags(hparams)
  return hparams


def _terminate_eval():
  tf.logging.info('Timeout passed with no new checkpoints ... terminating eval')
  return True


def _get_next_checkpoint():
  return tf.contrib.training.checkpoints_iterator(
      FLAGS.model_dir,
      timeout=FLAGS.eval_timeout,
      timeout_fn=_terminate_eval)


def _set_or_add_hparam(hparams, name, value):
  if getattr(hparams, name, None) is None:
    hparams.add_hparam(name, value)
  else:
    hparams.set_hparam(name, value)


def _load_global_step_from_checkpoint_dir(checkpoint_dir):
  try:
    checkpoint_reader = tf.train.NewCheckpointReader(
        tf.train.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0

def prepare_tf_config():
  hvd.init()
  ip_list = FLAGS.worker_hosts.split(',')
  worker_hosts = []
  for i, ip in enumerate(ip_list):
    worker_hosts.append(ip + ":" + str(4000 + i * 1000 + hvd.local_rank()))
  if len(worker_hosts) > 1:
    cluster = {"chief": [worker_hosts[0]],
               "worker": worker_hosts[1:]}
  else:
    cluster = {"chief": [worker_hosts[0]]}

  if FLAGS.task_index == 0:
    os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster,
         'task': {'type': "chief", 'index': 0}})
  else:
    os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster,
         'task': {'type': 'worker',
                  'index': FLAGS.task_index - 1}})

def main(_):
  mode = FLAGS.mode
  data_dir = FLAGS.data_dir
  model_dir = FLAGS.model_dir
  hparams = build_hparams()

  estimator_parmas = {}

  train_steps_per_epoch = int(
      math.ceil(hparams.num_train_images / float(hparams.train_batch_size)))
  eval_steps = hparams.num_eval_images // hparams.eval_batch_size
  eval_batch_size = (None if mode == 'train' else
                     hparams.eval_batch_size)

  model = slice_model_lib.AmoebaNetEstimatorModel(hparams, model_dir)

  save_checkpoints_steps = (FLAGS.save_checkpoints_steps or
                            FLAGS.iterations_per_loop)
  prepare_tf_config()
#  rewrite_options = rewriter_config_pb2.RewriterConfig(
#      layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF)
#  graph_options = config_pb2.GraphOptions(rewrite_options=rewrite_options)
  session_config = tf.ConfigProto(
#          graph_options=graph_options,
          allow_soft_placement=True,
          log_device_placement=False,
          gpu_options=tf.GPUOptions(allow_growth=True))
  if FLAGS.cross_pipeline:
    cluster_manager = cluster_utils.get_cluster_manager(config_proto=session_config)
  run_config = tf.estimator.RunConfig(
       log_step_count_steps=100,
       session_config=session_config,
       save_checkpoints_steps=save_checkpoints_steps)
  image_classifier = tf.estimator.Estimator(
      model_fn=model.model_fn,
      config=run_config,
      params=estimator_parmas)

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  imagenet_train = model_lib.InputPipeline(
      is_training=True, data_dir=data_dir, hparams=hparams)
  imagenet_eval = model_lib.InputPipeline(
      is_training=False, data_dir=data_dir, hparams=hparams)

  if hparams.moving_average_decay < 1:
    eval_hooks = [model_lib.LoadEMAHook(model_dir,
                                        hparams.moving_average_decay)]
  else:
    eval_hooks = []

  if mode == 'eval':
    for checkpoint in _get_next_checkpoint():
      tf.logging.info('Starting to evaluate.')
      try:
        eval_results = image_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            hooks=eval_hooks,
            checkpoint_path=checkpoint)
        tf.logging.info('Evaluation results: %s' % eval_results)
      except tf.errors.NotFoundError:
        # skip checkpoint if it gets deleted prior to evaluation
        tf.logging.info('Checkpoint %s no longer exists ... skipping')
  elif mode == 'train_and_eval':
    current_step = _load_global_step_from_checkpoint_dir(model_dir)
    tf.logging.info('Starting training at step=%d.' % current_step)
    train_steps_per_eval = int(
        hparams.num_epochs_per_eval * train_steps_per_epoch)
    # Final Evaluation if training is finished.
    if current_step >= hparams.num_epochs * train_steps_per_epoch:
      eval_results = image_classifier.evaluate(
          input_fn=imagenet_eval.input_fn, steps=eval_steps, hooks=eval_hooks)
      tf.logging.info('Evaluation results: %s' % eval_results)
    while current_step < hparams.num_epochs * train_steps_per_epoch:
      image_classifier.train(
          input_fn=imagenet_train.input_fn, steps=train_steps_per_eval)
      current_step += train_steps_per_eval
      tf.logging.info('Starting evaluation at step=%d.' % current_step)
      eval_results = image_classifier.evaluate(
          input_fn=imagenet_eval.input_fn, steps=eval_steps, hooks=eval_hooks)
      tf.logging.info('Evaluation results: %s' % eval_results)
  elif mode == 'predict':
    for checkpoint in _get_next_checkpoint():
      tf.logging.info('Starting prediction ...')
      time_hook = model_lib.SessionTimingHook()
      eval_hooks.append(time_hook)
      result_iter = image_classifier.predict(
          input_fn=imagenet_eval.input_fn,
          hooks=eval_hooks,
          checkpoint_path=checkpoint,
          yield_single_examples=False)
      results = list(itertools.islice(result_iter, eval_steps))
      tf.logging.info('Inference speed = {} images per second.'.format(
          time_hook.compute_speed(len(results) * eval_batch_size)))
  elif mode == 'train':
    current_step = _load_global_step_from_checkpoint_dir(model_dir)
    total_step = int(hparams.num_epochs * train_steps_per_epoch)
    if current_step < total_step:
      tf.logging.info('Starting training ...')
      image_classifier.train(
          input_fn=imagenet_train.input_fn,
          steps=min(total_step-current_step, FLAGS.max_steps))
  else:
    tf.logging.info('Mode not found.')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.DEBUG)
  app.run(main)
