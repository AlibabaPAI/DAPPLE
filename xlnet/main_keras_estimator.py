from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import json
import os

import numpy as np

from datasets.squad_tfrecords import input_fn_builder
from models.xlnet import build_xlnet_for_keras_estimator
from utils import get_run_config
from utils import load_model_weights_from_original_checkpoint

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

flags.DEFINE_string("distribution", default="",
                    help="distribution strategy") ## mirrored or collective
flags.DEFINE_string('worker_hosts', '', 'IP address and port list')
flags.DEFINE_integer('task_index', 0, 'worker or ps task index')
flags.DEFINE_integer('display', 100,
                     """set for train loss print info.""")

FLAGS = flags.FLAGS

def prepare_tf_config():
  if FLAGS.distribution != 'mirrored':
    worker_hosts = FLAGS.worker_hosts.split(',')
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
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['HOROVOD_HIERARCHICAL_ALLREDUCE'] = '1'

## Disable Tensor Fusion
#os.environ['HOROVOD_FUSION_THRESHOLD']='800000000'
os.environ['HVD_DEBUG']='1'
def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    prepare_tf_config()
    model = build_xlnet_for_keras_estimator(embedding_dim=FLAGS.embedding_dim,
                                            num_token=FLAGS.num_token,
                                            num_layer=FLAGS.num_layer,
                                            num_head=FLAGS.num_head,
                                            feed_forward_dim=FLAGS.feed_forward_dim,
                                            attention_head_dim=FLAGS.attention_head_dim,
                                            target_len=FLAGS.target_len,
                                            dropout=FLAGS.dropout,
                                            is_training=True,
                                            attention_dropout=FLAGS.dropatt)
#    load_model_weights_from_original_checkpoint(
#        model,
#        num_layer=FLAGS.num_layer,
#        checkpoint_path=FLAGS.init_checkpoint)

    # Only TensorFlow native optimizers are supported with DistributionStrategy.
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                                   epsilon=FLAGS.adam_epsilon))

    estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                      config=get_run_config(FLAGS))

    #estimator.train(input_fn=input_fn_builder(input_glob=FLAGS.train_file,
    #                                          batch_size=FLAGS.train_batch_size,
    #                                          is_training=True),
    #                max_steps=FLAGS.train_steps)

    #print ("variable shape")
    #print (np.sum([np.prod(v.get_shape().as_list()) for v in tf.global_variables()]))
    #print (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    train_input_fn = input_fn_builder(input_glob=FLAGS.train_file,
                                      batch_size=FLAGS.train_batch_size,
                                      is_training=True)
    if FLAGS.distribution == 'collective':
      print ("estimator.train_and_evaluate")
      train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
      #eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn)
      tf.estimator.train_and_evaluate(estimator, train_spec, None)
      #tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else:
      print ("estimator.train")
      estimator.train(input_fn=train_input_fn,
                      max_steps=FLAGS.train_steps)


if __name__ == "__main__":
    tf.app.run()
