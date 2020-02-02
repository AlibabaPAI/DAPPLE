"""Class for benchmarking a cnn network."""

import tensorflow as tf
from tensorflow.python.client import device_lib

from datasets.dataset_factory import dataset_factory
from datasets import dataset_utils
import graph_helper
from utils.hooks import hooks_helper
from losses.loss_factory import loss_factory
from train_flags import FLAGS

import horovod.tensorflow as hvd

class Graph(object):
  def __init__(self, target=None, num_workers=None, hparams=None, global_step=None):
    self._target = target
    self._num_workers = num_workers
    self._hparams = hparams
    self._global_step = global_step
    self._network_fn = None
    self._ds_iterator = None
    self._dataset_obj = None
    self._optimizer = None
    self._lrate = None
    self._loss_obj = None
    self._loss_fn = None
    self._accuracy = None
    self._train_op = None
    self._hooks = None
    if FLAGS.enable_pipeline:
      self._samples_per_step = FLAGS.batch_size * FLAGS.pipeline_micro_batch_num
    else:
      self._samples_per_step = FLAGS.batch_size

    self._build_graph()

  def _config_ds_iterator(self):
    with tf.device('/cpu:0'):
      # split dataset by worker
      data_sources = dataset_utils.get_tfrecord_files(self._num_workers, FLAGS.file_pattern)

      # select the preprocessing func
      dataset_obj_initializer = dataset_factory(FLAGS.dataset_name).get_dataset_obj()
      self._dataset_obj = dataset_obj_initializer(data_sources,
                                                  None,
                                                  None)
      self._ds_iterator = self._dataset_obj.get_dataset()

  def _config_learning_rate(self):
    self._lrate = graph_helper.configure_learning_rate(FLAGS.num_sample_per_epoch, self._global_step)

  def _config_optimizer(self):
    # if not self._lrate:
    #    raise ValueError('learning rate is None.')
    self._optimizer = graph_helper.configure_optimizer(self._lrate)


  def _get_loss_fn(self):
    """Specify the loss_fn which is defined by loss_name.

    See 'losses/default.py' for more information.
    """
    loss_obj_initializer = loss_factory(FLAGS.loss_name).get_loss_obj()
    self._loss_obj = loss_obj_initializer(self._ds_iterator, self._network_fn, self._hparams)
    self._loss_fn = self._loss_obj._get_loss_fn()
    if FLAGS.enable_pipeline:
      self._stage_fn = self._loss_obj._get_stages()
      return

    # whether wrapper with paisoar
    self._loss_fn = self._loss_fn()

  def _mean_accuracy(self):
    self._accuracy = tf.reduce_mean(self._loss_obj._get_accuracy())

  def _train_op_initializer(self):
    """Specify the train_op for forward and backward."""
    if FLAGS.enable_pipeline:
      if not FLAGS.cross_pipeline:
        all_devices = device_lib.list_local_devices()
        self.slice_devices = [v.name for v in all_devices if v.device_type=="GPU"]
      else:
        self.slice_devices = ["/job:worker/replica:0/task:0/device:GPU:%d" % hvd.local_rank(), "/job:worker/replica:0/task:1/device:GPU:%d" % hvd.local_rank()]
      self.slice_num = FLAGS.pipeline_device_num
      num_all_device = len(self.slice_devices)
      if num_all_device < self.slice_num:
        tf.logging.warning(" Local GPU device amount {} is less than assigned number {}"
                           " for pipeline.".format(num_all_device, self.slice_num))
        self.slice_num = num_all_device
      self.slice_devices = self.slice_devices[0:self.slice_num]
      tf.logging.info(" Slicing devices: {}".format(self.slice_devices))
      micro_batch_num = FLAGS.pipeline_micro_batch_num
      tf.logging.info(" PIPELINE enabled, model_slices = {}, micro_batches = {}."
                      .format(self.slice_num, micro_batch_num))

      losses = []
      all_outputs = []
      loss, outputs = self._stage_fn(slice_devices=self.slice_devices, dep_outputs=None)
      losses.append(loss)
      all_outputs.append(outputs)

      # NOTE(zycao): Never use [[[]]*n]*m to construct this empty list, shallow list would be created.
      layer_grads = [[[] for i in xrange(self.slice_num)] for j in xrange(micro_batch_num)]
      layer_vars = [[] for i in xrange(self.slice_num)]
      remained_vars = tf.trainable_variables()
      ys = losses[0]
      prev_grads=None
      # layers-1 ~ 1 compute grads
      for i in xrange(self.slice_num - 1, 0, -1):
        vars_i = [v for v in remained_vars if v.device==self.slice_devices[i]]
        remained_vars = [v for v in remained_vars if v not in vars_i]
        prev_y = all_outputs[0][i-1]
        y_grads = tf.gradients(ys=ys, xs=[prev_y]+vars_i, grad_ys=prev_grads, colocate_gradients_with_ops=True)
        ys = prev_y
        prev_grads = y_grads[0]
        grads_i = y_grads[1:]
        layer_grads[0][i] = [g for g in grads_i if g is not None]
        layer_vars[i] = [v for (g, v) in zip(grads_i, vars_i) if g is not None]
      # layer 0 compute grads
      grads_0 = tf.gradients(ys=ys, xs=remained_vars, grad_ys=prev_grads, colocate_gradients_with_ops=True)
      layer_grads[0][0] = [g for g in grads_0 if g is not None]
      layer_vars[0] = [v for (g, v) in zip(grads_0, remained_vars) if g is not None]

      # other micro_batch_num
      for j in xrange(1, micro_batch_num):
        dep_outputs = []
        for i in xrange(self.slice_num):
          dep_outputs.append(all_outputs[j-1][i] if i+j < self.slice_num else layer_grads[i+j-self.slice_num][i])
        loss, outputs = self._stage_fn(slice_devices=self.slice_devices, dep_outputs=dep_outputs)
        losses.append(loss)
        all_outputs.append(outputs)
        ys = losses[j]
        prev_grads=None
        for i in xrange(self.slice_num - 1, 0, -1):
          prev_y = all_outputs[j][i-1]
          y_grads = tf.gradients(ys=ys, xs=[prev_y]+layer_vars[i], grad_ys=prev_grads, colocate_gradients_with_ops=True)
          ys = prev_y
          prev_grads = y_grads[0]
          grads_i = y_grads[1:]
          layer_grads[j][i] = [g for g in grads_i if g is not None]
        grads_0 = tf.gradients(ys=ys, xs=layer_vars[0], grad_ys=prev_grads, colocate_gradients_with_ops=True)
        layer_grads[j][0] = [g for g in grads_0 if g is not None]

      grads_set = []
      vars_set = []
      for i in xrange(self.slice_num):
        for j in xrange(len(layer_grads[0][i])):
          grad_i_set = [layer_grads[m][i][j] for m in range(micro_batch_num)]
          #print (grad_i_set)
          if micro_batch_num == 1:
            with tf.device(self.slice_devices[i]):
              acc_grads = grad_i_set[0]
          else:
            with tf.control_dependencies(grad_i_set), tf.device(self.slice_devices[i]):
              if isinstance(grad_i_set[0], tf.IndexedSlices):
                acc_grads = tf.add_n(grad_i_set)
              else:
                acc_grads = tf.accumulate_n(grad_i_set)
          grads_set.append(acc_grads)
          vars_set.append(layer_vars[i][j])
      if FLAGS.max_gradient_norm is not None:
        (clipped_grads, _) = tf.clip_by_global_norm([tf.cast(g, tf.float32) for g in grads_set],
                                                    clip_norm=FLAGS.max_gradient_norm)
        grads_set = [tf.cast(c, g.dtype) for (c, g) in zip(clipped_grads, grads_set)]
      grads_and_vars = zip(grads_set, vars_set)
      self._loss_fn = losses[-1]
      gradients_list = [[] for i in xrange(len(self.slice_devices))]
      for grad, var in grads_and_vars:
        for i in xrange(len(self.slice_devices)):
          if var.device == self.slice_devices[i]:
            gradients_list[i].append((grad, var))
            break
      avg_grads_and_vars = []
      for i in xrange(len(self.slice_devices)):
        with tf.device(self.slice_devices[i]):
          for grad, var in gradients_list[i]:
            if isinstance(grad, tf.IndexedSlices):
              grad = tf.convert_to_tensor(grad)
            avg_grad = hvd.allreduce(grad)
            avg_grads_and_vars.append((avg_grad, var))
      grads_and_vars = avg_grads_and_vars
    else:
      """Specify the train_op for forward and backward."""
      grads_and_vars = self._optimizer.compute_gradients(self._loss_fn, colocate_gradients_with_ops=True)
      gvs = [(g, v) for (g, v) in grads_and_vars if g is not None]
      grads_set = [g for g, _ in gvs]
      vars_set = [v for _, v in gvs]
      if FLAGS.max_gradient_norm is not None:
        (clipped_grads, _) = tf.clip_by_global_norm([tf.cast(g, tf.float32) for g in grads_set],
                                                    clip_norm=FLAGS.max_gradient_norm)
        grads_set = [tf.cast(c, g.dtype) for (c, g) in zip(clipped_grads, grads_set)]
      grads_and_vars = zip(grads_set, vars_set)

    self._train_op = self._optimizer.apply_gradients(grads_and_vars, global_step=self._global_step)

  def _load_checkpoint(self):
    if FLAGS.model_dir:
      from utils import util
      util.load_checkpoint()

  def _trainable_variables_logger(self):
    """Print trainable variables info, including name and size."""
    if FLAGS.print_model_statistics:
      graph_helper.print_model_statistics()

  def _specific_hooks(self):
    """Specify hooks used for training or evaluation.
        Users could re-implement to alter hooks.

    See 'utils/hooks/hooks_helper.py' for more information.
    """
    params = dict()
    if FLAGS.log_loss_every_n_iters > 0:
      params['tensors_to_log'] = {'loss': self.display_loss(), 'accuracy': self._accuracy, 'lrate:': self._lrate}
      params['samples_per_step'] = self._samples_per_step
    self._hooks = hooks_helper.get_train_hooks(params)

  def _build_graph(self):
    """Build the graph."""
    self._config_ds_iterator()
    self._config_learning_rate()
    self._config_optimizer()
    self._get_loss_fn()
    self._mean_accuracy()
    self._train_op_initializer()
    self._load_checkpoint()

    self._trainable_variables_logger()
    self._specific_hooks()

  def run(self):
    tf.logging.info('training starts.')
    print('hooks:', self._hooks)
    with tf.train.MonitoredTrainingSession(
        self._target,
        is_chief=(FLAGS.task_index == 0),
        hooks=self._hooks) as sess:
      tf.train.global_step(sess, self._global_step)
      try:
        while not sess.should_stop():
          sess.run(self._train_op)
      except tf.errors.OutOfRangeError:
        print('All threads done.')
      except Exception as e:
        import sys, traceback
        tf.logging.error(e.message)
        traceback.print_exc(file=sys.stdout)
    tf.logging.info('training ends.')

  def run_eval(self):
    pass

  def display_loss(self):
    return self._loss_fn if isinstance(self._loss_fn, tf.Tensor) else self._loss_fn.replicas[0]
