import collections
import re

import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from utils import cluster_utils
import horovod.tensorflow as hvd
import pdb

def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def get_run_config(FLAGS):
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=64,
        inter_op_parallelism_threads=64,
        gpu_options=tf.GPUOptions(allow_growth=True,
                                  force_gpu_compatible=True,
                                  per_process_gpu_memory_fraction=1.0))
    # 24 layers: OOM, 2 layers: slower
#    session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # MemOptType
#    session_config.graph_options.rewrite_options.memory_optimization=rewriter_config_pb2.RewriterConfig.RECOMPUTATION_HEURISTICS
    # default "gradients/"
#    session_config.graph_options.rewrite_options.memory_optimizer_target_node_name_scope="Recomp"
    strategy = None
    if FLAGS.distribution == 'mirrored':
      print("MirroredStrategy")
      strategy = tf.contrib.distribute.MirroredStrategy(
          num_gpus=FLAGS.num_core_per_host, all_dense=True)
    elif FLAGS.distribution == 'collective':
      print("CollectiveAllReduceStrategy")
      strategy = tf.contrib.distribute.CollectiveAllReduceStrategy(
           num_gpus_per_worker=FLAGS.num_core_per_host,
           cross_tower_ops_type='horovod',
           all_dense=True)
    else:
     print("strategy is None")
     strategy=None

    hvd.init()
    if FLAGS.cross_pipeline:
      cluster_manager = cluster_utils.get_cluster_manager(config_proto=session_config)
    run_config = tf.estimator.RunConfig(
        #model_dir=FLAGS.model_dir,
        session_config=session_config,
        #keep_checkpoint_max=FLAGS.max_save,
        save_checkpoints_secs=None,
        save_checkpoints_steps=None, ## QQ: disable checkpoints
        #save_checkpoints_steps=FLAGS.save_steps,
        train_distribute=strategy,
    )
    return run_config


def load_model_weights_from_original_checkpoint(model,
                                                num_layer,
                                                checkpoint_path):
    loader = checkpoint_loader(checkpoint_path)
    model.get_layer(name='Embed-Token').set_weights([
        loader('model/transformer/word_embedding/lookup_table'),
    ])

    model.get_layer(name='Embed-Segment').set_weights([
        loader('model/transformer/seg_embed')
    ])

    model.get_layer(name='Relative-Bias').set_weights([
        loader('model/transformer/r_w_bias'),
        loader('model/transformer/r_r_bias'),
        loader('model/transformer/r_s_bias')
    ])

    for i in range(num_layer):
        att_kernel_name = 'model/transformer/layer_{}/rel_attn/{}/kernel'
        model.get_layer(name='Attention-{}'.format(i + 1)).set_weights([
            loader(att_kernel_name.format(i, 'q')),
            loader(att_kernel_name.format(i, 'k')),
            loader(att_kernel_name.format(i, 'v')),
            loader(att_kernel_name.format(i, 'r')),
            loader(att_kernel_name.format(i, 'o')),
        ])

        model.get_layer(name='Attention-Normal-{}'.format(i + 1)).set_weights([
            loader('model/transformer/layer_{}/rel_attn/LayerNorm/gamma'.format(i)),
            loader('model/transformer/layer_{}/rel_attn/LayerNorm/beta'.format(i)),
        ])
        model.get_layer(name='FeedForward-{}'.format(i + 1)).set_weights([
            loader('model/transformer/layer_{}/ff/layer_1/kernel'.format(i)),
            loader('model/transformer/layer_{}/ff/layer_1/bias'.format(i)),
            loader('model/transformer/layer_{}/ff/layer_2/kernel'.format(i)),
            loader('model/transformer/layer_{}/ff/layer_2/bias'.format(i)),
        ])

        model.get_layer(name='FeedForward-Normal-{}'.format(i + 1)).set_weights([
            loader('model/transformer/layer_{}/ff/LayerNorm/gamma'.format(i)),
            loader('model/transformer/layer_{}/ff/LayerNorm/beta'.format(i)),
        ])


def get_assignment_map_from_checkpoint(tvars, num_layer):
    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    assignment_map = collections.OrderedDict()

    assignment_map['model/transformer/word_embedding/lookup_table'] \
        = name_to_variable['Embed-Token/embeddings']

    assignment_map['model/transformer/seg_embed'] \
        = name_to_variable['Embed-Segment/seg_emb']

    assignment_map['model/transformer/r_w_bias'] = \
        name_to_variable['Relative-Bias/bias_context']

    assignment_map['model/transformer/r_r_bias'] = \
        name_to_variable['Relative-Bias/bias_relative']

    assignment_map['model/transformer/r_s_bias'] = \
        name_to_variable['Relative-Bias/bias_segment']

    att_kernel_name = 'model/transformer/layer_{}/rel_attn/{}/kernel'

    for i in range(num_layer):
        assignment_map[att_kernel_name.format(i, 'q')] = \
            name_to_variable['Attention-{}/q'.format(i + 1)]
        assignment_map[att_kernel_name.format(i, 'k')] = \
            name_to_variable['Attention-{}/k'.format(i + 1)]
        assignment_map[att_kernel_name.format(i, 'v')] = \
            name_to_variable['Attention-{}/v'.format(i + 1)]
        assignment_map[att_kernel_name.format(i, 'r')] = \
            name_to_variable['Attention-{}/r'.format(i + 1)]
        assignment_map[att_kernel_name.format(i, 'o')] = \
            name_to_variable['Attention-{}/o'.format(i + 1)]

        assignment_map['model/transformer/layer_{}/rel_attn/LayerNorm/gamma'.format(i)] = \
            name_to_variable['Attention-Normal-{}/gamma'.format(i + 1)]
        assignment_map['model/transformer/layer_{}/rel_attn/LayerNorm/beta'.format(i)] = \
            name_to_variable['Attention-Normal-{}/beta'.format(i + 1)]

        assignment_map['model/transformer/layer_{}/ff/layer_1/kernel'.format(i)] = \
            name_to_variable['FeedForward-{}/FeedForward-{}_W1'.format(i + 1, i + 1)]
        assignment_map['model/transformer/layer_{}/ff/layer_1/bias'.format(i)] = \
            name_to_variable['FeedForward-{}/FeedForward-{}_b1'.format(i + 1, i + 1)]
        assignment_map['model/transformer/layer_{}/ff/layer_2/kernel'.format(i)] = \
            name_to_variable['FeedForward-{}/FeedForward-{}_W2'.format(i + 1, i + 1)]
        assignment_map['model/transformer/layer_{}/ff/layer_2/bias'.format(i)] = \
            name_to_variable['FeedForward-{}/FeedForward-{}_b2'.format(i + 1, i + 1)]

        assignment_map['model/transformer/layer_{}/ff/LayerNorm/gamma'.format(i)] = \
            name_to_variable['FeedForward-Normal-{}/gamma'.format(i + 1)]
        assignment_map['model/transformer/layer_{}/ff/LayerNorm/beta'.format(i)] = \
            name_to_variable['FeedForward-Normal-{}/beta'.format(i + 1)]

    return assignment_map


def init_from_checkpoint(FLAGS, global_vars=False):
    tvars = tf.global_variables() if global_vars else tf.trainable_variables()

    init_checkpoint = FLAGS.init_checkpoint
    tf.logging.info("Initialize from the ckpt {}".format(init_checkpoint))

    assignment_map = get_assignment_map_from_checkpoint(tvars, num_layer=FLAGS.num_layer)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


def create_optimizer(FLAGS):
    global_step = tf.train.get_or_create_global_step()

    warmup_lr = (tf.cast(global_step, tf.float32)
                 / tf.cast(FLAGS.warmup_steps, tf.float32)
                 * FLAGS.learning_rate)

    decay_lr = tf.train.polynomial_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        end_learning_rate=FLAGS.learning_rate * FLAGS.min_lr_ratio)

    learning_rate = tf.where(global_step < 1000,
                             warmup_lr, decay_lr)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        epsilon=FLAGS.adam_epsilon)

    return optimizer
