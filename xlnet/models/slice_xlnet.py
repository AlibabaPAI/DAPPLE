import tensorflow as tf

from activations import gelu
from layers import FeedForward
from layers import LayerNormalization
from layers import RelativeBias
from layers import RelativeMultiHeadAttention
from layers import RelativeSegmentEmbedding
from layers import get_attn_mask, get_pos_emb
from losses import XLnetLoss
from utils import backend as K
from utils import keras

from utils import cluster_utils
from tensorflow.python.platform import flags
import numpy as np
FLAGS = flags.FLAGS


class XlnetSlice(object):

  def __init__(
        self,
        num_token,
        num_layer,
        num_head,
        embedding_dim,
        attention_head_dim,
        feed_forward_dim,
        target_len,
        is_training,
        memory_len=None,
        dropout=0.0,
        attention_dropout=0.0,
        attention_type=None,
        shared_biases=True):
    self.num_layer = num_layer
    self.dropout = dropout
    self.attention_dropout = attention_dropout

    self.token_embed = keras.layers.Embedding(input_dim=num_token,
                                         output_dim=embedding_dim,
                                         name='Embed-Token')

    initializer = keras.initializers.get('normal')
    initializer.__setattr__("stddev", 0.02)

    self.segment_ids_trans = keras.layers.Lambda(lambda x: K.transpose(x))
    self.segment_mat_embed = RelativeSegmentEmbedding(
        num_layer=num_layer,
        num_head=num_head,
        attention_dim=attention_head_dim,
        initializer=initializer,
        name='Embed-Segment')

    self.relative_bias = RelativeBias(
        num_layer=num_layer,
        num_head=num_head,
        attention_head_dim=attention_head_dim,
        bias_initializer=initializer,
        name='Relative-Bias')

    self.attention = []
    self.attention_add = []
    self.attention_layer_norm = []
    self.feed_forward = []
    self.feed_forward_add = []
    self.feed_forward_layer_norm = []
    for i in range(num_layer):
        self.attention.append(RelativeMultiHeadAttention(
            num_head=num_head,
            attention_head_dim=attention_head_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            dropatt=attention_dropout,
            is_training=is_training,
            initializer=initializer,
            name='Attention-{}'.format(i + 1),
        ))

        self.attention_add.append(tf.keras.layers.Add(name='Attention-Residual-{}'.format(i + 1)))
        self.attention_layer_norm.append(LayerNormalization(name='Attention-Normal-{}'.format(i + 1)))

        self.feed_forward.append(FeedForward(
            feed_forward_dim=feed_forward_dim,
            embedding_dim=embedding_dim,
            dropout_rate=dropout,
            kernel_initializer=initializer,
            activation=gelu,
            name='FeedForward-{}'.format(i + 1)
        ))
        self.feed_forward_add.append(tf.keras.layers.Add(name='FeedForward-Residual-{}'.format(i + 1)))
        self.feed_forward_layer_norm.append(LayerNormalization(name='FeedForward-Normal-{}'.format(i + 1)))
    self.xlnet_loss = XLnetLoss(d_model=embedding_dim,
                             seq_len=target_len,
                             kernel_initializer=initializer,
                             name="XLNET_LOSS")


  def build(self, inputs, dep_outputs=None, is_training=True):
    # pipeline num device
    devices = cluster_utils.get_pipeline_devices(FLAGS.pipeline_device_num)
    ndev = len(devices)
    # embedding + dropout + ... + dropout
    nstage = self.num_layer + 3
    def calc_device(i):
      # original stage fn
      idx = int((i+2)/((nstage+1) / ndev + 1))
      split_layer_id = 13 if FLAGS.num_layer == 24 else 19
      # For XLNet-24:
      # stage fn 1: Forward-11 in stage0
      # stage fn 2: Forward-10 in stage0
      # For XLNet-36:
      # stage fn: Forward 17 in stage0
      # 13:(11:13) for xlnet-24
      # 19:(17:19) for xlnet-36
      if i < split_layer_id:
        return 0
      else:
        return 1
      return idx

    dep = None
    device_idx = 0
    if dep_outputs is not None and dep_outputs[device_idx] is not None:
      dep = dep_outputs[device_idx] \
            if isinstance(dep_outputs[device_idx], list) else [dep_outputs[device_idx]]
    with tf.control_dependencies(dep), tf.device(devices[device_idx]):
      input_ids, input_mask, segment_ids, cls_index, \
      p_mask, start_positions, end_positions, is_impossible = inputs

      # 1MB, [512,512,1,1]
      attn_mask = get_attn_mask(input_mask)

      input_ids_trans = keras.layers.Lambda(lambda x: K.transpose(x))(input_ids)
      token_embed = self.token_embed(input_ids_trans)
      segment_ids_trans = self.segment_ids_trans(segment_ids)
      # 2MB [512,512,1,2] 192KB [24, 2, 16, 64]
      segment_mat, segment_embed = self.segment_mat_embed(segment_ids_trans)
      # 3*96KB [24, 16, 64]
      r_w_bias, r_r_bias, r_s_bias = self.relative_bias(input_ids_trans)
      token_embed_dropout = keras.layers.Dropout(rate=self.dropout,
                                                 name='Embed-Token-Dropout'
                                                 )(token_embed, training=is_training)
      content_output = token_embed_dropout
      pos_emb = get_pos_emb([input_ids_trans, token_embed])
      # 4MB [1024, 1, 1024]
      pos_emb = keras.layers.Dropout(rate=self.dropout
                                   )(pos_emb, training=is_training)
      if FLAGS.short_cut_fake:
        attn_mask = tf.constant(1.0, shape=[512, 512, 1, 1], dtype=np.float32)
        segment_mat = tf.constant(1.0, shape=[512, 512, 1, 2], dtype=np.float32)
        pos_emb = tf.constant(1.0, shape=[1024, 1, 1024], dtype=np.float32)
      if FLAGS.short_cut_fuse:
        attn_mask_flat = tf.reshape(attn_mask, [-1])
        segment_mat_flat = tf.reshape(segment_mat, [-1])
        segment_embed_flat = tf.reshape(segment_embed, [-1])
        pos_emb_flat = tf.reshape(pos_emb, [-1])
        r_w_bias_flat = tf.reshape(r_w_bias, [-1])
        r_r_bias_flat = tf.reshape(r_r_bias, [-1])
        r_s_bias_flat = tf.reshape(r_s_bias, [-1])
        fused = tf.concat([attn_mask_flat, segment_mat_flat, segment_embed_flat, \
                  pos_emb_flat, r_w_bias_flat, r_r_bias_flat, r_s_bias_flat], 0)
    def _build_output(query, i, pos_emb, segment_mat, segment_embed, \
                      attn_mask, r_w_bias, r_r_bias, r_s_bias):
        segment_embed_i = keras.layers.Lambda(lambda x: x[i])(segment_embed)
        r_w_bias_i = keras.layers.Lambda(lambda x: x[i])(r_w_bias)
        r_r_bias_i = keras.layers.Lambda(lambda x: x[i])(r_r_bias)
        r_s_bias_i = keras.layers.Lambda(lambda x: x[i])(r_s_bias)
        attention_input = query
        _output = self.attention[i](
            [query, pos_emb, segment_embed_i, segment_mat,
             r_w_bias_i, r_r_bias_i, r_s_bias_i, attn_mask])
        _output = self.attention_add[i]([attention_input, _output])
        _output = self.attention_layer_norm[i](_output)
        feed_forward_input = keras.layers.Lambda(lambda x: K.identity(x))(_output)
        _output = self.feed_forward[i](_output, training=is_training)
        _output = self.feed_forward_add[i]([feed_forward_input, _output])
        _output = self.feed_forward_layer_norm[i](_output)
        return _output

    # output list of all stages
    stage_outputs = []
    # previous output, for the first stage, it is input_ids
    prev_output = input_ids
    # previous device index, init value is 0
    prev_device_idx = 0
    for i in range(self.num_layer):
        layer = i + 2
        device_idx = calc_device(layer)
        dep = None
        boundary = False
        if device_idx != prev_device_idx:
          # current layer cross the stage
          boundary = True
          if dep_outputs is not None and dep_outputs[device_idx] is not None:
            dep = dep_outputs[device_idx] \
                  if isinstance(dep_outputs[device_idx], list) else [dep_outputs[device_idx]]
          stage_outputs.append(prev_output)
          prev_device_idx = device_idx
        with tf.control_dependencies(dep), tf.device(devices[device_idx]):
          if boundary:
            if FLAGS.short_cut_fake:
              attn_mask = tf.constant(1.0, shape=[512, 512, 1, 1], dtype=np.float32)
              segment_mat = tf.constant(1.0, shape=[512, 512, 1, 2], dtype=np.float32)
              pos_emb = tf.constant(1.0, shape=[1024, 1, 1024], dtype=np.float32)
            if FLAGS.short_cut_fuse:
              num_layers = FLAGS.num_layer
              attn_mask_flat, segment_mat_flat, segment_embed_flat, \
                pos_emb_flat, r_w_bias_flat, r_r_bias_flat, r_s_bias_flat = \
                  tf.split(fused, [512*512*1, 512*512*2, num_layers*2*1024, \
                           1024*1024, num_layers*1024, num_layers*1024, num_layers*1024], 0)
              attn_mask = tf.reshape(attn_mask_flat, [512, 512, 1, 1])
              segment_mat = tf.reshape(segment_mat_flat, [512, 512, 1, 2])
              segment_embed = tf.reshape(segment_embed_flat, [num_layers, 2, 16, 64])
              pos_emb = tf.reshape(pos_emb_flat, [1024, 1, 1024])
              r_w_bias = tf.reshape(r_w_bias_flat, [num_layers, 16, 64])
              r_r_bias = tf.reshape(r_r_bias_flat, [num_layers, 16, 64])
              r_s_bias = tf.reshape(r_s_bias_flat, [num_layers, 16, 64])
              print(attn_mask, segment_mat, segment_embed, pos_emb, r_w_bias, r_r_bias, r_s_bias)
          content_output = _build_output(content_output, i, pos_emb, segment_mat, \
                          segment_embed, attn_mask, r_w_bias, r_r_bias, r_s_bias)
        prev_output = content_output

    # current layer cross the stage
    layer = self.num_layer + 2
    device_idx = calc_device(layer)
    dep = None
    if device_idx != prev_device_idx:
      # current layer cross the stage
      if dep_outputs is not None and dep_outputs[device_idx] is not None:
        dep = dep_outputs[device_idx] \
              if isinstance(dep_outputs[device_idx], list) else [dep_outputs[device_idx]]
      stage_outputs.append(prev_output)
    with tf.control_dependencies(dep), tf.device(devices[device_idx]):
      output = keras.layers.Dropout(rate=self.dropout)(content_output, training=is_training)

      xlnet_loss = self.xlnet_loss([cls_index,
                                    start_positions,
                                    end_positions,
                                    is_impossible,
                                    p_mask,
                                    output])
      stage_outputs.append(xlnet_loss)

    return xlnet_loss, stage_outputs
