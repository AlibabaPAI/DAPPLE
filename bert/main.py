"""Startup script for TensorFlow.

See the README for more information.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils.platform import cluster_utils
from graph import Graph
from train_flags import FLAGS
from utils.misc import hparams_utils

import horovod.tensorflow as hvd

def create_config_proto():
  """Returns session config proto."""
  config = tf.ConfigProto(
    log_device_placement=FLAGS.log_device_placement,
    inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
    intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(
      force_gpu_compatible=True,
      allow_growth=True))
  config.graph_options.enable_bfloat16_sendrecv = FLAGS.enable_bfloat16_sendrecv
  return config


def main(_):
  hparams = hparams_utils.create_hparams(FLAGS) if FLAGS.create_hparams else None
  if FLAGS.use_grad_checkpoint:
    from utils.accelerator import memory_saving_gradients
    memory_saving_gradients.use_grad_checkpoint()

  hvd.init()
  cluster_manager = cluster_utils.get_cluster_manager(config_proto=create_config_proto())
  with tf.device(cluster_manager.device_exp()):
    global_step = tf.train.get_or_create_global_step()

    graph = Graph(cluster_manager.get_target(), cluster_manager.num_workers(), hparams, global_step)

    if FLAGS.task_type == 'train':
      graph.run()
    else:
      raise ValueError('task_type [%s] was not recognized' % FLAGS.task_type)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.app.run()

