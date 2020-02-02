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
from tensorflow.python.platform import flags

import numpy as np
FLAGS = flags.FLAGS

def build_xlnet_for_keras_estimator(num_token,
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
    input_ids = keras.layers.Input(shape=(target_len,), name='input_ids', dtype=tf.int32)
    input_mask = keras.layers.Input(shape=(target_len,), name='input_mask', dtype=tf.float32)
    segment_ids = keras.layers.Input(shape=(target_len,), name='segment_ids', dtype=tf.int32)
    cls_index = keras.layers.Input(shape=(1,), name='cls_index', dtype=tf.int32)
    p_mask = keras.layers.Input(shape=(target_len,), name='p_mask', dtype=tf.float32)
    start_positions = keras.layers.Input(shape=(1,), name='start_positions', dtype=tf.int32)
    end_positions = keras.layers.Input(shape=(1,), name='end_positions', dtype=tf.int32)
    is_impossible = keras.layers.Input(shape=(1,), name='is_impossible', dtype=tf.float32)

    inputs = [input_ids, input_mask, segment_ids, cls_index,
              p_mask, start_positions, end_positions, is_impossible]

    xlnet_loss = build_xlnet_for_tf_estimator(inputs=inputs,
                                              num_token=num_token,
                                              num_layer=num_layer,
                                              num_head=num_head,
                                              embedding_dim=embedding_dim,
                                              attention_head_dim=attention_head_dim,
                                              feed_forward_dim=feed_forward_dim,
                                              target_len=target_len,
                                              is_training=is_training,
                                              memory_len=None,
                                              dropout=dropout,
                                              attention_dropout=attention_dropout,
                                              attention_type=attention_type,
                                              shared_biases=shared_biases
                                              )

    model = tf.keras.Model(inputs=inputs, outputs=xlnet_loss)

    return model


def build_xlnet_for_tf_estimator(
        inputs,
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
    input_ids, input_mask, segment_ids, cls_index, \
    p_mask, start_positions, end_positions, is_impossible = inputs

    attn_mask = get_attn_mask(input_mask)

    input_ids_trans = keras.layers.Lambda(lambda x: K.transpose(x))(input_ids)
    token_embed = keras.layers.Embedding(input_dim=num_token,
                                         output_dim=embedding_dim,
                                         name='Embed-Token')(input_ids_trans)
    token_embed_dropout = keras.layers.Dropout(rate=dropout,
                                               name='Embed-Token-Dropout'
                                               )(token_embed, training=is_training)

    pos_emb = get_pos_emb([input_ids_trans, token_embed])
    pos_emb = keras.layers.Dropout(rate=dropout
                                   )(pos_emb, training=is_training)

    initializer = keras.initializers.get('normal')
    initializer.__setattr__("stddev", 0.02)

    segment_ids_trans = keras.layers.Lambda(lambda x: K.transpose(x))(segment_ids)
    segment_mat, segment_embed = RelativeSegmentEmbedding(
        num_layer=num_layer,
        num_head=num_head,
        attention_dim=attention_head_dim,
        initializer=initializer,
        name='Embed-Segment',
    )(segment_ids_trans)

    r_w_bias, r_r_bias, r_s_bias = RelativeBias(
        num_layer=num_layer,
        num_head=num_head,
        attention_head_dim=attention_head_dim,
        bias_initializer=initializer,
        name='Relative-Bias',
    )(input_ids_trans)

    content_output = token_embed_dropout
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

    for i in range(num_layer):
        attention = RelativeMultiHeadAttention(
            num_head=num_head,
            attention_head_dim=attention_head_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            dropatt=attention_dropout,
            is_training=is_training,
            initializer=initializer,
            name='Attention-{}'.format(i + 1),
        )

        attention_add = tf.keras.layers.Add(name='Attention-Residual-{}'.format(i + 1))
        attention_layer_norm = LayerNormalization(name='Attention-Normal-{}'.format(i + 1))

        feed_forward = FeedForward(
            feed_forward_dim=feed_forward_dim,
            embedding_dim=embedding_dim,
            dropout_rate=dropout,
            kernel_initializer=initializer,
            activation=gelu,
            name='FeedForward-{}'.format(i + 1)
        )
        feed_forward_add = tf.keras.layers.Add(name='FeedForward-Residual-{}'.format(i + 1))
        feed_forward_layer_norm = LayerNormalization(name='FeedForward-Normal-{}'.format(i + 1))

        segment_embed_i = keras.layers.Lambda(lambda x: x[i])(segment_embed)
        r_w_bias_i = keras.layers.Lambda(lambda x: x[i])(r_w_bias)
        r_r_bias_i = keras.layers.Lambda(lambda x: x[i])(r_r_bias)
        r_s_bias_i = keras.layers.Lambda(lambda x: x[i])(r_s_bias)
        if FLAGS.short_cut_fuse:
          attn_mask_flat, segment_mat_flat, segment_embed_flat, \
            pos_emb_flat, r_w_bias_flat, r_r_bias_flat, r_s_bias_flat = \
              tf.split(fused, [512*512*1, 512*512*2, 24*2*1024, \
                       1024*1024, 24*1024, 24*1024, 24*1024], 0)
          attn_mask = tf.reshape(attn_mask_flat, [512, 512, 1, 1])
          segment_mat = tf.reshape(segment_mat_flat, [512, 512, 1, 2])
          segment_embed = tf.reshape(segment_embed_flat, [24, 2, 16, 64])
          pos_emb = tf.reshape(pos_emb_flat, [1024, 1, 1024])
          r_w_bias = tf.reshape(r_w_bias_flat, [24, 16, 64])
          r_r_bias = tf.reshape(r_r_bias_flat, [24, 16, 64])
          r_s_bias = tf.reshape(r_s_bias_flat, [24, 16, 64])
          print(attn_mask, segment_mat, segment_embed, pos_emb, r_w_bias, r_r_bias, r_s_bias)

        def _build_output(query):
            attention_input = query
            _output = attention(
                [query, pos_emb, segment_embed_i, segment_mat,
                 r_w_bias_i, r_r_bias_i, r_s_bias_i, attn_mask])
            _output = attention_add([attention_input, _output])
            _output = attention_layer_norm(_output)
            feed_forward_input = keras.layers.Lambda(lambda x: K.identity(x))(_output)
            _output = feed_forward(_output, training=is_training)
            _output = feed_forward_add([feed_forward_input, _output])
            _output = feed_forward_layer_norm(_output)
            return _output

        content_output = _build_output(content_output)

    output = keras.layers.Dropout(rate=dropout)(content_output, training=is_training)

    xlnet_loss = XLnetLoss(d_model=embedding_dim,
                           seq_len=target_len,
                           kernel_initializer=initializer,
                           name="XLNET_LOSS")([cls_index,
                                               start_positions,
                                               end_positions,
                                               is_impossible,
                                               p_mask,
                                               output])

    return xlnet_loss
