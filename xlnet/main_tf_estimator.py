from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets.squad_tfrecords import input_fn_builder
from models.slice_xlnet import XlnetSlice
from utils import create_optimizer, init_from_checkpoint
from utils import get_run_config
from utils import cluster_utils
import horovod.tensorflow as hvd
import json
import os

flags = tf.app.flags

# Model
flags.DEFINE_integer("embedding_dim", default=None,
                     help="Hidden embedding dimensions throughout the model")
flags.DEFINE_integer("num_token", default=None,
                     help="Number of distinct tokens")
flags.DEFINE_integer("num_layer", default=None,
                     help="Number of basic encoder layers")
flags.DEFINE_integer("num_head", default=None,
                     help="Number of heads for attention")
flags.DEFINE_integer("feed_forward_dim", default=None,
                     help="Dimension inside position-wise feed-forward layer")
flags.DEFINE_integer("attention_head_dim", default=None,
                     help="Dimension of each attention head")
flags.DEFINE_integer("target_len", default=None,
                     help="The length of prediction block")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")

# Training
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
flags.DEFINE_string("train_file", default="",
                    help="Path of train file.")
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                         "Could be a pretrained model or a finetuned model.")
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
                          "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_integer("train_batch_size", default=48,
                     help="batch size for training")
flags.DEFINE_integer("train_steps", default=8000,
                     help="Number of training steps")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("max_save", default=5,
                     help="Max number of checkpoints to save. "
                          "Use 0 to save all.")
flags.DEFINE_float("learning_rate", default=3e-5, help="initial learning rate")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="Adam epsilon")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_float("lr_layer_decay_rate", default=0.75,
                   help="Top layer: lr[L] = FLAGS.learning_rate."
                        "Lower layers: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_string("distribution", default="",
                    help="distribution strategy") ## mirrored or collective
flags.DEFINE_string('worker_hosts', '', 'IP address and port list')
flags.DEFINE_string('job_name', '', 'worker or ps')
flags.DEFINE_integer('task_index', 0, 'worker or ps task index')
flags.DEFINE_bool('cross_pipeline', False, 'cross pipeline')
flags.DEFINE_integer('pipeline_device_num', 1, 'device num in one pipeline')
flags.DEFINE_integer('micro_batch_num', 1, 'num of batches in one pipeline')
flags.DEFINE_bool('global_clip', False, 'global_clip')
flags.DEFINE_bool('short_cut_fake', False, 'short cut fake')
flags.DEFINE_bool('short_cut_fuse', False, 'short cut fuse')

FLAGS = flags.FLAGS

def prepare_tf_config():
  hvd.init()
  if FLAGS.distribution != 'mirrored':
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

def get_train_op(FLAGS, grads_and_vars):
    global_step = tf.train.get_or_create_global_step()
    optimizer = create_optimizer(FLAGS)
#    grads_and_vars = optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
    gradients, variables = zip(*grads_and_vars)
    if FLAGS.global_clip:
      clipped, gnorm = tf.clip_by_global_norm(gradients, FLAGS.clip)
      if getattr(FLAGS, "lr_layer_decay_rate", 1.0) != 1.0:
          for i in range(len(clipped)):
              for l in range(FLAGS.num_layer):
                  if "Attention-{}/".format(l + 1) in variables[i].name or \
                          "Attention-Normal-{}/".format(l + 1) in variables[i].name or \
                          "FeedForward-{}/".format(l + 1) in variables[i].name or \
                          "FeedForward-Normal-{}/".format(l + 1) in variables[i].name:
                      abs_rate = FLAGS.lr_layer_decay_rate ** (FLAGS.num_layer - 1 - l)
                      clipped[i] *= abs_rate
                      # tf.logging.info("Apply mult {:.4f} to layer-{} grad of {}".format(abs_rate, l, variables[i].name))
                      break
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

    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
#        zip(clipped, variables), global_step=global_step)

    return train_op


def get_model_fn():
    def model_fn(features, labels, mode, params):
        inputs = [features['input_ids'], features['input_mask'], features['segment_ids'],
                  features['cls_index'], features['p_mask'], features['start_positions'],
                  features['end_positions'], features['is_impossible']]

        slice_xlnet = XlnetSlice(
            embedding_dim=FLAGS.embedding_dim,
            num_token=FLAGS.num_token,
            num_layer=FLAGS.num_layer,
            num_head=FLAGS.num_head,
            feed_forward_dim=FLAGS.feed_forward_dim,
            attention_head_dim=FLAGS.attention_head_dim,
            target_len=FLAGS.target_len,
            dropout=FLAGS.dropout,
            is_training=True,
            attention_dropout=FLAGS.dropatt)

        total_loss, outputs = slice_xlnet.build(inputs)

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
          for i in xrange(slice_num):
            dep_outputs.append(all_outputs[j-1][i] if i+j < slice_num else layer_grads[i+j-slice_num][i])
          loss, outputs = slice_xlnet.build(inputs, dep_outputs=dep_outputs)
          losses.append(loss)
          all_outputs.append(outputs)
          ys = losses[j]
          prev_grads=None
          for i in xrange(slice_num - 1, 0, -1):
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
        for i in xrange(slice_num):
          for j in xrange(len(layer_grads[0][i])):
            grad_i_set = [layer_grads[m][i][j] for m in range(micro_batch_num)]
            #print (grad_i_set)
            if micro_batch_num == 1:
              with tf.device(devices[i]):
                acc_grads = grad_i_set[0]
            else:
              with tf.control_dependencies(grad_i_set), tf.device(devices[i]):
                if isinstance(grad_i_set[0], tf.IndexedSlices):
                  acc_grads = tf.add_n(grad_i_set)
                else:
                  acc_grads = tf.accumulate_n(grad_i_set)
            grads_set.append(acc_grads)
            vars_set.append(layer_vars[i][j])
        grads_and_vars = zip(grads_set, vars_set)

#        init_from_checkpoint(FLAGS)
        train_op = get_train_op(FLAGS, grads_and_vars)

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op, )

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    prepare_tf_config()

    estimator = tf.estimator.Estimator(
        model_fn=get_model_fn(),
        config=get_run_config(FLAGS))

    estimator.train(input_fn=input_fn_builder(input_glob=FLAGS.train_file,
                                              batch_size=FLAGS.train_batch_size,
                                              is_training=True),
                    max_steps=FLAGS.train_steps)


if __name__ == "__main__":
    tf.app.run()
