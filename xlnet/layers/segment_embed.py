import tensorflow as tf

from utils import backend as K
from utils import keras


class RelativeSegmentEmbedding(keras.layers.Layer):

    def __init__(self,
                 num_layer,
                 num_head,
                 attention_dim,
                 initializer='zeros',
                 **kwargs):
        self.num_layer = num_layer
        self.num_head = num_head
        self.attention_dim = attention_dim
        self.initializer = keras.initializers.get(initializer)
        super(RelativeSegmentEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.seg_emb = None

    def build(self, input_shape):
        self.seg_emb = self.add_weight(
            shape=(self.num_layer, 2, self.num_head, self.attention_dim),
            initializer=self.initializer,
            dtype=K.floatx(),
            name='seg_emb',
        )
        super(RelativeSegmentEmbedding, self).build(input_shape)

    def call(self, inputs):
        batch_size = K.shape(inputs)[1]
        seg_id = inputs
        # Convert `seg_id` to one-hot `seg_mat`
        # mem_pad = tf.zeros([mlen, bsz], dtype=tf.int32)
        mem_pad = keras.layers.Lambda(lambda x: tf.zeros([0, x], dtype=tf.int32))(batch_size)
        seg_id = K.cast(seg_id, dtype=tf.int32)

        # cat_ids = tf.concat([mem_pad, seg_id], 0)
        cat_ids = K.concatenate([mem_pad, seg_id], 0)

        # `1` indicates not in the same segment [qlen x klen x bsz] 512, 512, 3
        """
        seg_mat = tf.cast(
            tf.logical_not(tf.equal(seg_id[:, None], cat_ids[None, :])),
            tf.int32)
        """
        tmp0 = keras.layers.Lambda(lambda x: tf.equal(seg_id[:, None], cat_ids[None, :]))([seg_id, cat_ids])
        tmp = keras.layers.Lambda(lambda x: tf.logical_not(x))(tmp0)
        seg_mat = K.cast(tmp, tf.int32)

        seg_mat = K.one_hot(seg_mat, 2)
        return [seg_mat, K.identity(self.seg_emb)]

    def get_config(self):
        config = {
            'num_layer':self.num_layer,
            'num_head': self.num_head,
            'attention_dim': self.attention_dim,
            'initializer': keras.initializers.serialize(self.initializer),

        }
        base_config = super(RelativeSegmentEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
