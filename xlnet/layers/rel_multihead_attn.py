import tensorflow as tf

from lambda_funcs import rel_shift
from utils import backend as K
from utils import keras


class RelativeMultiHeadAttention(keras.layers.Layer):

    def __init__(self,
                 num_head,
                 attention_head_dim,
                 embedding_dim,
                 dropout,
                 dropatt,
                 is_training,
                 initializer='zeros',
                 **kwargs):
        self.num_head = num_head
        self.attention_head_dim = attention_head_dim
        self.embedding_dim = embedding_dim
        self.dropatt = dropatt
        self.dropout = dropout
        self.is_training = is_training
        self.supports_masking = True
        self.initializer = keras.initializers.get(initializer)
        self.q_head_weight = None
        self.k_head_weight = None
        self.v_head_weight = None
        self.r_head_weight = None
        self.proj_o = None

        super(RelativeMultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # assert isinstance(input_shape, list)
        # content heads
        self.q_head_weight = self.add_weight(
            shape=(self.embedding_dim, self.num_head, self.attention_head_dim),
            initializer=self.initializer,
            dtype=K.floatx(),
            name='q',
        )
        self.k_head_weight = self.add_weight(
            shape=(self.embedding_dim, self.num_head, self.attention_head_dim),
            initializer=self.initializer,
            dtype=K.floatx(),
            name='k',
        )
        self.v_head_weight = self.add_weight(
            shape=(self.embedding_dim, self.num_head, self.attention_head_dim),
            initializer=self.initializer,
            dtype=K.floatx(),
            name='v',
        )
        self.r_head_weight = self.add_weight(
            shape=(self.embedding_dim, self.num_head, self.attention_head_dim),
            initializer=self.initializer,
            dtype=K.floatx(),
            name='r',
        )
        self.proj_o = self.add_weight(
            shape=(self.embedding_dim, self.num_head, self.attention_head_dim),
            initializer=self.initializer,
            dtype=K.floatx(),
            name='o',
        )
        super(RelativeMultiHeadAttention, self).build(input_shape)

    @staticmethod
    def _rel_attn_core(q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                       r_w_bias, r_r_bias, r_s_bias, attn_mask, dropatt, is_training,
                       scale):
        """Core relative positional attention operations."""

        # content based attention score
        # ac = tf.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)
        ac = keras.layers.Lambda(lambda x: tf.einsum('ibnd,jbnd->ijbn', x[0] + x[1], x[2]))(
            [q_head, r_w_bias, k_head_h])

        # position based attention score
        # bd = tf.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
        bd = keras.layers.Lambda(lambda x: tf.einsum('ibnd,jbnd->ijbn', x[0] + x[1], x[2]))(
            [q_head, r_r_bias, k_head_r])

        tmp_shape = keras.layers.Lambda(lambda x: tf.shape(x)[1])(ac)
        bd = rel_shift([bd, tmp_shape])

        # segment based attention score
        # ef = tf.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed)
        ef = keras.layers.Lambda(lambda x: tf.einsum('ibnd,snd->ibns', x[0] + x[1], x[2]))(
            [q_head, r_s_bias, seg_embed])

        # ef = tf.einsum('ijbs,ibns->ijbn', seg_mat, ef)
        ef = keras.layers.Lambda(lambda x: tf.einsum('ijbs,ibns->ijbn', x[0], x[1]))([seg_mat, ef])

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * scale
        # attn_score = keras.layers.Lambda(lambda x: (x[0] + x[1] + x[2]) * x[3])([ac, bd, ef, scale])

        # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
        attn_score = attn_score - 1e30 * attn_mask
        # attn_score = keras.layers.Lambda(lambda x: x[0] - x[1] * x[2])([attn_score, 1e30, attn_mask])

        # attention probability
        # attn_prob = tf.nn.softmax(attn_score, 1)
        attn_prob = keras.layers.Lambda(lambda x: tf.nn.softmax(x, 1))(attn_score)

        # attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
        attn_prob = keras.layers.Dropout(rate=dropatt)(attn_prob, training=is_training)

        # attention output
        # attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)
        attn_vec = keras.layers.Lambda(lambda x: tf.einsum('ijbn,jbnd->ibnd', x[0], x[1]))([attn_prob, v_head_h])
        return attn_vec

    def call(self, inputs):
        # assert isinstance(inputs, list)
        scale = 1 / (self.attention_head_dim ** 0.5)
        #attention_input( 512, ?, 1024)
        #pos_emb ()
        #attn_mask(512,512,?,1)
        attention_input, pos_emb, seg_embed, seg_mat, r_w_bias, r_r_bias, r_s_bias, attn_mask = inputs

        # q_head_h = tf.einsum('ibh,hnd->ibnd', h, self.q_head_weight)
        q_head_h = keras.layers.Lambda(lambda x:
                                       tf.einsum('ibh,hnd->ibnd', x[0], x[1]))([attention_input, self.q_head_weight])
        # k_head_h = tf.einsum('ibh,hnd->ibnd', h, self.k_head_weight)
        k_head_h = keras.layers.Lambda(lambda x:
                                       tf.einsum('ibh,hnd->ibnd', x[0], x[1]))([attention_input, self.k_head_weight])

        # v_head_h = tf.einsum('ibh,hnd->ibnd', h, self.v_head_weight)
        v_head_h = keras.layers.Lambda(lambda x:
                                       tf.einsum('ibh,hnd->ibnd', x[0], x[1]))([attention_input, self.v_head_weight])

        # k_head_r = tf.einsum('ibh,hnd->ibnd', r, self.r_head_weight)
        k_head_r = keras.layers.Lambda(lambda x:
                                       tf.einsum('ibh,hnd->ibnd', x[0], x[1]))([pos_emb, self.r_head_weight])

        # core attention ops
        attn_vec = self._rel_attn_core(
            q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
            r_r_bias, r_s_bias, attn_mask, self.dropatt, self.is_training, scale)

        # post processing
        attn_out = keras.layers.Lambda(lambda x: tf.einsum('ibnd,hnd->ibh', x[0], x[1]))(
            [attn_vec, K.identity(self.proj_o)])
        attn_out = keras.layers.Dropout(rate=self.dropout)(attn_out, training=self.is_training)
        return attn_out

    def get_config(self):
        config = {
            'num_head': self.num_head,
            'attention_head_dim': self.attention_head_dim,
            'embedding_dim': self.embedding_dim,
            'dropout': self.dropout,
            'dropatt': self.dropatt,
            'is_training': self.is_training,
            'initializer': keras.initializers.serialize(self.initializer),
        }
        base_config = super(RelativeMultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
