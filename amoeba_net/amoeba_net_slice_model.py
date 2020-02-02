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
"""AmoebaNet ImageNet model functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from common import cluster_utils
import horovod.tensorflow as hvd

import inception_preprocessing
import model_builder_slice as model_builder
import model_specs

# Random cropping constants
_RESIZE_SIDE_MIN = 300
_RESIZE_SIDE_MAX = 600

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

FLAGS = flags.FLAGS

def imagenet_hparams():
  """Returns default ImageNet training params.

  These defaults are for full training. For search training, some should be
  modified to increase the speed of the search.
  """
  return tf.contrib.training.HParams(
      ##########################################################################
      # Input pipeline params. #################################################
      ##########################################################################

      image_size=299,
      num_train_images=1281167,
      num_eval_images=50000,
      num_label_classes=1001,
      ##########################################################################
      # Architectural params. ##################################################
      ##########################################################################

      # The total number of regular cells (summed across all stacks). Reduction
      # cells are not included.
      num_cells=18,
      reduction_size=256,
      stem_reduction_size=32,

      # How many reduction cells to use between the stacks of regular cells.
      num_reduction_layers=2,

      # Stem.
      stem_type='imagenet',  # 'imagenet' or others
      num_stem_cells=2,  # 2 if stem_type == 'imagenet' else 0

      # Implementation details.
      data_format='NCHW',  # 'NHWC' or 'NCHW'.

      ##########################################################################
      # Training params. #######################################################
      ##########################################################################

      # Summed across all TPU cores training a model.
      train_batch_size=32,

      num_epochs=100.,

      # Auxiliary head.
      use_aux_head=True,
      aux_scaling=0.4,

      # Regularization.
      l1_decay_rate=0.0,
      label_smoothing=0.1,
      drop_connect_keep_prob=0.7,
      # `drop_connect_version` determines how the drop_connect probabilites are
      # set/increased over time:
      # -v1: increase dropout probability over training,
      # -v2: increase dropout probability as you increase the number of cells,
      #      so the top cell has the highest dropout and the lowest cell has the
      #      lowest dropout,
      # -v3: Do both v1 and v2.
      drop_connect_version='v1',
      drop_path_burn_in_steps=0,
      # `drop_connect_condition` determines under what conditions drop_connect
      # is used:
      # -identity: Dropout all paths except identity connections,
      # -all: Dropout all paths,
      # -separable: Dropout only paths containing a separable conv operation.
      dense_dropout_keep_prob=0.5,
      batch_norm_epsilon=0.001,
      batch_norm_decay=0.9997,
      shuffle_buffer=20000,

      # Any value <= 0 means it is unused
      gradient_clipping_by_global_norm=10.0,

      # Learning rate schedule.
      lr=0.015,
      lr_decay_method='exponential',
      lr_decay_value=0.97,
      lr_num_epochs_per_decay=2.4,
      lr_warmup_epochs=3.0,
      weight_decay=4e-05,

      # Optimizer.
      optimizer='rmsprop',  # 'sgd', 'mom', 'adam' or 'rmsprop'
      rmsprop_decay=0.9,
      rmsprop_momentum_rate=0.9,
      rmsprop_epsilon=1.0,
      momentum_rate=0.9,
      use_nesterov=1,

      ##########################################################################
      # Eval and reporting params. #############################################
      ##########################################################################

      # This number should be a multiple of the number of TPU shards
      # used for eval (e.g., 2 for a 1x1 or 8 for a 2x2).
      eval_batch_size=40,

      # How many different crops are fed into one model. Also affects training.
      num_input_images=1,

      moving_average_decay=0.9999,

      write_summaries=0,

      ##########################################################################
      # Other params. ##########################################################
      ##########################################################################
      num_shards=None,
      distributed_group_size=1,
      use_tpu=False)


def build_hparams(cell_name='amoeba_net_d'):
  """Build tf.Hparams for training Amoeba Net.

  Args:
    cell_name:         Which of the cells in model_specs.py to use to build the
                       amoebanet neural network; the cell names defined in that
                       module correspond to architectures discovered by an
                       evolutionary search described in
                       https://arxiv.org/abs/1802.01548.

  Returns:
    A set of tf.HParams suitable for Amoeba Net training.
  """
  hparams = imagenet_hparams()
  operations, hiddenstate_indices, used_hiddenstates = (
      model_specs.get_normal_cell(cell_name))
  hparams.add_hparam('normal_cell_operations', operations)
  hparams.add_hparam('normal_cell_hiddenstate_indices',
                     hiddenstate_indices)
  hparams.add_hparam('normal_cell_used_hiddenstates',
                     used_hiddenstates)
  operations, hiddenstate_indices, used_hiddenstates = (
      model_specs.get_reduction_cell(cell_name))
  hparams.add_hparam('reduction_cell_operations',
                     operations)
  hparams.add_hparam('reduction_cell_hiddenstate_indices',
                     hiddenstate_indices)
  hparams.add_hparam('reduction_cell_used_hiddenstates',
                     used_hiddenstates)
  hparams.set_hparam('data_format', 'NHWC')
  return hparams


def formatted_hparams(hparams):
  """Formatts the hparams into a readable string.

  Also looks for attributes that have not correctly been added to the hparams
  and prints the keys as "bad keys". These bad keys may be left out of iterators
  and cirumvent type checking.

  Args:
    hparams: an HParams instance.

  Returns:
    A string.
  """
  # Look for bad keys (see docstring).
  good_keys = set(hparams.values().keys())
  bad_keys = []
  for key in hparams.__dict__:
    if key not in good_keys and not key.startswith('_'):
      bad_keys.append(key)
  bad_keys.sort()

  # Format hparams.
  readable_items = [
      '%s: %s' % (k, v) for k, v in sorted(hparams.values().iteritems())]
  readable_items.append('Bad keys: %s' % ','.join(bad_keys))
  readable_string = ('\n'.join(readable_items))
  return readable_string


class AmoebaNetEstimatorModel(object):
  """Definition of AmoebaNet."""

  def __init__(self, hparams, model_dir):
    self.hparams = hparams
    self.model_dir = model_dir

  def _calc_num_trainable_params(self):
    self.num_trainable_params = np.sum([
        np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()
    ])
    tf.logging.info(
        'number of trainable params: {}'.format(self.num_trainable_params))

    devices = cluster_utils.get_pipeline_devices(FLAGS.pipeline_device_num)
    for i in xrange(len(devices)):
      vars_i = [v for v in tf.trainable_variables() if v.device==devices[i]]
      num_trainable_params_i = np.sum([
          np.prod(var.get_shape().as_list()) for var in vars_i
      ])
      tf.logging.info(
          'number of trainable params: {}'.format(num_trainable_params_i))
#    for v in tf.trainable_variables():
#      var_size = np.prod(v.get_shape().as_list())
#      tf.logging.info(
#          '{}: {}'.format(v.name, var_size))




  def _build_learning_rate_schedule(self, global_step):
    """Build learning rate."""
    steps_per_epoch = (
        self.hparams.num_train_images // self.hparams.train_batch_size)
    lr_warmup_epochs = 0
    if self.hparams.lr_decay_method == 'exponential':
      lr_warmup_epochs = self.hparams.lr_warmup_epochs
    learning_rate = model_builder.build_learning_rate(
        self.hparams.lr,
        self.hparams.lr_decay_method,
        global_step,
        total_steps=steps_per_epoch * self.hparams.num_epochs,
        decay_steps=steps_per_epoch * self.hparams.lr_num_epochs_per_decay,
        decay_factor=self.hparams.lr_decay_value,
        add_summary=False,
        warmup_steps=int(lr_warmup_epochs * steps_per_epoch))

    learning_rate = tf.maximum(
        learning_rate, 0.0001 * self.hparams.lr, name='learning_rate')
    return learning_rate

  def _build_network(self, features, labels, mode, dep_outputs=None):
    devices = cluster_utils.get_pipeline_devices(FLAGS.pipeline_device_num)
    """Build a network that returns loss and logits from features and labels."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    steps_per_epoch = float(
        self.hparams.num_train_images) / self.hparams.train_batch_size
    num_total_steps = int(steps_per_epoch * self.hparams.num_epochs)
    self.hparams.set_hparam('drop_path_burn_in_steps', num_total_steps)

    hparams = copy.deepcopy(self.hparams)
    if not is_training:
      hparams.set_hparam('use_aux_head', False)
      hparams.set_hparam('weight_decay', 0)
      hparams.set_hparam('use_bp16', False)

    tf.logging.info(
        'Amoeba net received hparams for {}:\n{}'.format(
            'training' if is_training else 'eval',
            formatted_hparams(hparams)))
    device_idx = 0
    dep = None
    if dep_outputs is not None and dep_outputs[device_idx] is not None:
      dep = dep_outputs[device_idx] \
            if isinstance(dep_outputs[device_idx], list) else [dep_outputs[device_idx]]
    with tf.control_dependencies(dep), tf.device(devices[device_idx]):
      logits, end_points = model_builder.build_network(
        features, hparams.num_label_classes, is_training, hparams, dep_outputs)

    if is_training:
     with tf.device(devices[1]):
      labels = tf.one_hot(labels, hparams.num_label_classes)
      loss = model_builder.build_softmax_loss(
          logits,
          end_points,
          labels,
          label_smoothing=hparams.label_smoothing,
          add_summary=False)

    # Calculate and print the number of trainable parameters in the model
#    if is_training:
#      flops = model_builder.compute_flops_per_example(hparams.train_batch_size)
#    else:
#      flops = model_builder.compute_flops_per_example(hparams.eval_batch_size)
#    tf.logging.info('number of flops: {}'.format(flops))
    self._calc_num_trainable_params()
    stage_outputs = end_points['stage_outputs']
    stage_outputs = [stage_outputs]
    stage_outputs.append(loss)
    return loss, stage_outputs

  def _build_optimizer(self, learning_rate):
    """Build optimizer."""
    if self.hparams.optimizer == 'sgd':
      tf.logging.info('Using SGD optimizer')
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=learning_rate)
    elif self.hparams.optimizer == 'momentum':
      tf.logging.info('Using Momentum optimizer')
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=self.hparams.momentum_rate)
    elif self.hparams.optimizer == 'rmsprop':
      tf.logging.info('Using RMSProp optimizer')
      optimizer = tf.train.RMSPropOptimizer(
          learning_rate,
          RMSPROP_DECAY,
          momentum=RMSPROP_MOMENTUM,
          epsilon=RMSPROP_EPSILON)
    else:
      tf.logging.fatal('Unknown optimizer:', self.hparams.optimizer)

    return optimizer

  def _build_train_op(self, optimizer, grads_and_vars, global_step):
    """Build train_op from optimizer and loss."""
    if self.hparams.gradient_clipping_by_global_norm > 0.0:
      g, v = zip(*grads_and_vars)
      g, _ = tf.clip_by_global_norm(
          g, self.hparams.gradient_clipping_by_global_norm)
      grads_and_vars = zip(g, v)

      gradients, variables = zip(*grads_and_vars)
    if FLAGS.cross_pipeline and hvd.size() > 1:
      # pipeline num device
      devices = cluster_utils.get_pipeline_devices(FLAGS.pipeline_device_num)
      gradients_list = [[] for i in xrange(len(devices))]
      for grad, var in grads_and_vars:
        for i in xrange(len(devices)):
          if var.device == devices[i]:
            gradients_list[i].append((grad, var))
            break
      avg_grads_and_vars = []
      for i in xrange(len(devices)):
        with tf.device(devices[i]):
          for grad, var in gradients_list[i]:
            if isinstance(grad, tf.IndexedSlices):
              grad = tf.convert_to_tensor(grad)
            avg_grad = hvd.allreduce(grad)
            avg_grads_and_vars.append((avg_grad, var))
      grads_and_vars = avg_grads_and_vars

    return optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  def model_fn(self, features, labels, mode, params):
    """Build the model based on features, labels, and mode.

    Args:
      features: The features dictionary containing the data Tensor
        and the number of examples.
      labels: The labels Tensor resulting from calling the model.
      mode: A string indicating the training mode.
      params: A dictionary of hyperparameters.

    Returns:
      A tf.estimator.EstimatorSpec.
    """
    del params
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    if is_training:
      features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC
    total_loss, outputs = self._build_network(features, labels, mode)

    devices = cluster_utils.get_pipeline_devices(FLAGS.pipeline_device_num)
    slice_num = len(devices)
    micro_batch_num = FLAGS.micro_batch_num
    losses = []
    all_outputs = []
    losses.append(total_loss)
    all_outputs.append(outputs)
    layer_grads = [[[] for i in xrange(slice_num)] for j in xrange(micro_batch_num)]
    layer_vars = [[] for i in xrange(slice_num)]
    remained_vars = tf.trainable_variables()
    ys = losses[0]
    prev_grads=None
    # layers-1 ~ 1 compute grads
    for i in xrange(slice_num - 1, 0, -1):
      vars_i = [v for v in remained_vars if v.device==devices[i]]
      remained_vars = [v for v in remained_vars if v not in vars_i]
      prev_y = all_outputs[0][i-1]
      prev_y = prev_y if isinstance(prev_y, list) else [prev_y]
      num_tensors = len(prev_y)
      y_grads = tf.gradients(ys=ys, xs=prev_y+vars_i, grad_ys=prev_grads, colocate_gradients_with_ops=True)
      ys = prev_y
      prev_grads = y_grads[0:num_tensors]
      grads_i = y_grads[num_tensors:]
      layer_grads[0][i] = [g for g in grads_i if g is not None]
      layer_vars[i] = [v for (g, v) in zip(grads_i, vars_i) if g is not None]
    # layer 0 compute grads
    grads_0 = tf.gradients(ys=ys, xs=remained_vars, grad_ys=prev_grads, colocate_gradients_with_ops=True)
    layer_grads[0][0] = [g for g in grads_0 if g is not None]
    layer_vars[0] = [v for (g, v) in zip(grads_0, remained_vars) if g is not None]

    # other micro_batch_num
    for j in xrange(1, micro_batch_num):
      dep_outputs = []
      for i in xrange(slice_num):
        dep_outputs.append(all_outputs[j-1][i] if i+j < 2*slice_num-1 else layer_grads[i+j-2*slice_num+1][i])
      loss, outputs = self._build_network(features, labels, mode, dep_outputs=dep_outputs)
      losses.append(loss)
      all_outputs.append(outputs)
      ys = losses[j]
      prev_grads=None
      for i in xrange(slice_num - 1, 0, -1):
        prev_y = all_outputs[j][i-1]
        prev_y = prev_y if isinstance(prev_y, list) else [prev_y]
        num_tensors = len(prev_y)
        y_grads = tf.gradients(ys=ys, xs=prev_y+layer_vars[i], grad_ys=prev_grads, colocate_gradients_with_ops=True)
        ys = prev_y
        prev_grads = y_grads[0:num_tensors]
        grads_i = y_grads[num_tensors:]
        layer_grads[j][i] = [g for g in grads_i if g is not None]
      grads_0 = tf.gradients(ys=ys, xs=layer_vars[0], grad_ys=prev_grads, colocate_gradients_with_ops=True)
      layer_grads[j][0] = [g for g in grads_0 if g is not None]

    grads_set = []
    vars_set = []
    for i in xrange(slice_num):
      for j in xrange(len(layer_grads[0][i])):
        grad_i_set = [layer_grads[m][i][j] for m in range(micro_batch_num)]
        #print (grad_i_set)
        if micro_batch_num == 1:
          with tf.device(grad_i_set[0].device):
            acc_grads = grad_i_set[0]
        else:
          with tf.control_dependencies(grad_i_set), tf.device(grad_i_set[0].device): # replica
            if isinstance(grad_i_set[0], tf.IndexedSlices):
              acc_grads = tf.add_n(grad_i_set)
            else:
              acc_grads = tf.accumulate_n(grad_i_set)
        grads_set.append(acc_grads)
        vars_set.append(layer_vars[i][j])
    grads_and_vars = zip(grads_set, vars_set)
#######################

    train_op = None

    if is_training:
      global_step = tf.train.get_or_create_global_step()
      gs_t = tf.reshape(tf.cast(global_step, tf.int32), [1])

      # Setup learning rate schedule
      learning_rate = self._build_learning_rate_schedule(global_step)

      # Setup optimizer.
      optimizer = self._build_optimizer(learning_rate)

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(None): # original is update_ops
        train_op = self._build_train_op(optimizer, grads_and_vars,
                                        global_step=global_step)

      if self.hparams.moving_average_decay > 0:
        ema = tf.train.ExponentialMovingAverage(
            decay=self.hparams.moving_average_decay, num_updates=global_step)
        variables_to_average = (tf.trainable_variables() +
                                tf.moving_average_variables())
        with tf.control_dependencies([train_op]):
          with tf.name_scope('moving_average'):
            train_op = ema.apply(variables_to_average)

      lr_t = tf.reshape(learning_rate, [1])
      host_call = None
      if self.hparams.enable_hostcall:
        def host_call_fn(gs, lr):
          # Outfeed supports int32 but global_step is expected to be int64.
          gs = tf.cast(tf.reduce_mean(gs), tf.int64)
          with tf.contrib.summary.create_file_writer(
              self.model_dir).as_default():
            with tf.contrib.summary.always_record_summaries():
              tf.contrib.summary.scalar('learning_rate', tf.reduce_mean(lr),
                                        step=gs)
              return tf.contrib.summary.all_summary_ops()
        host_call = (host_call_fn, [gs_t, lr_t])

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op)


class InputPipeline(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

  Args:
    is_training: `bool` for whether the input is for training
  """

  def __init__(self, is_training, data_dir, hparams, eval_from_hub=False):
    self.is_training = is_training
    self.data_dir = data_dir
    self.hparams = hparams
    self.num_classes = 1001
    self.eval_from_hub = eval_from_hub

  def _dataset_parser(self, serialized_proto):
    """Parse an Imagenet record from value."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(dtype=tf.int64),
    }

    features = tf.parse_single_example(serialized_proto, keys_to_features)

    bbox = None

    image = features['image/encoded']
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = inception_preprocessing.preprocess_image(
        image=image,
        output_height=self.hparams.image_size,
        output_width=self.hparams.image_size,
        is_training=self.is_training,
        # If eval_from_hub, do not scale the images during preprocessing.
        scaled_images=not self.eval_from_hub,
        bbox=bbox)

    label = tf.cast(
        tf.reshape(features['image/class/label'], shape=[]), dtype=tf.int32)

    return image, label

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.

    Returns:
      A callable dataset object.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    if 'batch_size' in params:
      batch_size = params['batch_size']
    else:
      batch_size = (self.hparams.train_batch_size if self.is_training
                    else self.hparams.eval_batch_size)
    file_pattern = os.path.join(
        self.data_dir, 'train-*' if self.is_training else 'validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)
    if self.is_training:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=64, sloppy=True))
    dataset = dataset.shuffle(1024)

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            self._dataset_parser, batch_size=batch_size,
            num_parallel_batches=8, drop_remainder=True))

    if self.is_training:
      dataset = dataset.map(
          lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
          num_parallel_calls=8)

    dataset = dataset.prefetch(32)  # Prefetch overlaps in-feed with training
    return dataset  # Must return the dataset and not tensors for high perf!


class LoadEMAHook(tf.train.SessionRunHook):
  """Hook to load EMA into their corresponding variables."""

  def __init__(self, model_dir, moving_average_decay):
    super(LoadEMAHook, self).__init__()
    self._model_dir = model_dir
    self.moving_average_decay = moving_average_decay

  def begin(self):
    ema = tf.train.ExponentialMovingAverage(self.moving_average_decay)
    variables_to_restore = ema.variables_to_restore()
    self._load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
        tf.train.latest_checkpoint(self._model_dir), variables_to_restore)

  def after_create_session(self, sess, coord):
    tf.logging.info('Reloading EMA...')
    self._load_ema(sess)


class SessionTimingHook(tf.train.SessionRunHook):
  """Hook that computes speed based on session run time."""

  def __init__(self):
    # Lists of walltime.
    self._before_runs = []
    self._after_runs = []

  def before_run(self, run_context):
    self._before_runs.append(time.time())

  def after_run(self, run_context, results):
    self._after_runs.append(time.time())

  def compute_speed(self, num_samples):
    """Returns speed, in number of samples per second."""
    num_runs = len(self._before_runs)
    if num_runs == 0:
      raise ValueError('Session run time never recorded')
    if len(self._after_runs) != num_runs:
      raise ValueError(
          'Number of before_run events (%d) does not match '
          'number of after_run events (%d)' %
          (len(self._before_runs), len(self._after_runs)))
    total_eval_time = sum(self._after_runs[i] - self._before_runs[i]
                          for i in range(num_runs))
    if num_runs <= 1:
      tf.logging.warn(
          'Speed will be inaccurate with only one session run')
    else:
      # Exclude the first run, which tends to take much longer than other runs.
      total_eval_time -= (self._after_runs[0] - self._before_runs[0])
      # We assume num_samples are evenly distributed across runs.
      num_samples *= (float(num_runs - 1) / num_runs)
    return num_samples / max(total_eval_time, 1e-6)
