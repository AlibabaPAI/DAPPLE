from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets.squad_tfrecords import input_fn_builder
from models.xlnet import build_xlnet_for_tf_estimator
from utils import create_optimizer, init_from_checkpoint
from utils import get_run_config

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
flags.DEFINE_bool('cross_pipeline', False, 'cross pipeline')
flags.DEFINE_bool('short_cut_fake', False, 'short cut fake')
flags.DEFINE_bool('short_cut_fuse', False, 'short cut fuse')

FLAGS = flags.FLAGS


def get_train_op(FLAGS, loss):
    global_step = tf.train.get_or_create_global_step()
    optimizer = create_optimizer(FLAGS)
    grads_and_vars = optimizer.compute_gradients(loss)
    gradients, variables = zip(*grads_and_vars)
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

    train_op = optimizer.apply_gradients(
        zip(clipped, variables), global_step=global_step)

    return train_op


def get_model_fn():
    def model_fn(features, labels, mode, params):
        inputs = [features['input_ids'], features['input_mask'], features['segment_ids'],
                  features['cls_index'], features['p_mask'], features['start_positions'],
                  features['end_positions'], features['is_impossible']]

        total_loss = build_xlnet_for_tf_estimator(
            inputs=inputs,
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

#        init_from_checkpoint(FLAGS)
        train_op = get_train_op(FLAGS, total_loss)

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op, )

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    estimator = tf.estimator.Estimator(
        model_fn=get_model_fn(),
        config=get_run_config(FLAGS))

    estimator.train(input_fn=input_fn_builder(input_glob=FLAGS.train_file,
                                              batch_size=FLAGS.train_batch_size,
                                              is_training=True),
                    max_steps=FLAGS.train_steps)


if __name__ == "__main__":
    tf.app.run()
