import tensorflow as tf

from layers import LayerNormalization
from utils import backend as K
from utils import keras


class XLnetLoss(keras.layers.Layer):
    def __init__(self,
                 d_model,
                 seq_len,
                 kernel_initializer='normal',
                 **kwargs):
        super(XLnetLoss, self).__init__(**kwargs)
        self.supports_masking = True
        self.initializer = keras.initializers.get(kernel_initializer)
        self.max_seq_length = seq_len
        self.d_model = d_model
        self.dense = keras.layers.Dense(1, kernel_initializer=self.initializer)
        self.dense_0 = keras.layers.Dense(units=self.d_model,
                                        kernel_initializer=self.initializer,
                                        activation=keras.activations.tanh,
                                        name="dense_0")
        self.layer_norm = LayerNormalization()
        self.dense_1 = keras.layers.Dense(1, kernel_initializer=self.initializer, name="dense_1")
        self.dense_0_1 = keras.layers.Dense(
            self.d_model,
            activation=keras.activations.tanh,
            kernel_initializer=self.initializer, name="dense_0")
        self.dense_1_1 = keras.layers.Dense(
            1,
            kernel_initializer=self.initializer,
            name="dense_1",
            use_bias=False)

    def call(self, inputs, **kwargs):
        cls_index, start_positions, end_positions, is_impossible, p_mask, output = inputs
        # output 512, ?, 1024
        if len(start_positions.shape) == 1:
            start_positions = K.expand_dims(start_positions, axis=-1)
            cls_index = K.expand_dims(cls_index, axis=-1)
            end_positions = K.expand_dims(end_positions, axis=-1)
            is_impossible = K.expand_dims(is_impossible, axis=-1)

        cls_index = K.squeeze(cls_index, -1)
        start_positions = K.squeeze(start_positions, -1)
        end_positions = K.squeeze(end_positions, -1)
        is_impossible = K.squeeze(is_impossible, -1)

        # logit of the start position
        start_logits = self.dense(output)
        start_logits = K.transpose(K.squeeze(start_logits, -1))
        start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
        start_log_probs = keras.layers.Lambda(lambda x: tf.nn.log_softmax(x, -1))(start_logits_masked)

        # logit of the end position
        start_positions = K.cast(start_positions, dtype=tf.int32)
        # tart_index_1 = K.one_hot(start_positions, self.max_seq_length)
        start_index = keras.layers.Lambda(lambda x: tf.one_hot(x[0],
                                                               x[1], dtype=tf.float32))(
            [start_positions, self.max_seq_length])

        # start_features = tf.einsum("lbh,bl->bh", output, start_index)
        start_features = keras.layers.Lambda(lambda x: tf.einsum("lbh,bl->bh", x[0], x[1]))([output, start_index])
        start_features = K.expand_dims(start_features, 0)
        start_features = K.tile(start_features, [self.max_seq_length, 1, 1])
        tmp_concat = K.concatenate([output, start_features], axis=-1)
        end_logits = self.dense_0(tmp_concat)
        #end_logits = tf.contrib.layers.layer_norm(end_logits,begin_norm_axis=-1)
        end_logits = self.layer_norm(end_logits)

        end_logits = self.dense_1(end_logits)
        end_logits = K.transpose(K.squeeze(end_logits, -1))
        end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
        # end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
        end_log_probs = keras.layers.Lambda(lambda x: tf.nn.log_softmax(x, -1))(end_logits_masked)

        start_loss = - K.sum(start_log_probs * start_index, axis=-1)
        start_loss = K.mean(start_loss)
        end_positions = K.cast(end_positions, dtype=tf.int32)
        # end_index = K.one_hot(end_positions_squeeze, self.max_seq_length)
        end_index = keras.layers.Lambda(lambda x: tf.one_hot(x[0],
                                                             x[1], dtype=tf.float32))(
            [end_positions, self.max_seq_length])

        end_loss = - K.sum(end_log_probs * end_index, axis=-1)
        end_loss = K.mean(end_loss)
        total_loss = (start_loss + end_loss) * 0.5

        # an additional layer to predict answerability
        cls_index = K.cast(cls_index, dtype=tf.int32)
        # cls_index = K.one_hot(cls_index, self.max_seq_length)
        cls_index = keras.layers.Lambda(lambda x: tf.one_hot(x[0],
                                                             x[1], dtype=tf.float32))(
            [cls_index, self.max_seq_length])

        # cls_feature = tf.einsum("lbh,bl->bh", output, cls_index)
        cls_feature = keras.layers.Lambda(lambda x: tf.einsum("lbh,bl->bh", x[0], x[1]))([output, cls_index])
        # start_p = tf.nn.softmax(start_logits_masked, axis=-1, name="softmax_start")
        start_p = keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis=-1))(start_logits_masked)
        # start_feature = tf.einsum("lbh,bl->bh", output, start_p)
        start_feature = keras.layers.Lambda(lambda x:
                                            tf.einsum("lbh,bl->bh", x[0], x[1]))([output, start_p])

        # ans_feature = tf.concat([start_feature, cls_feature], -1)
        ans_feature = K.concatenate([start_feature, cls_feature], -1)
        ans_feature = self.dense_0_1(ans_feature)
        ans_feature = keras.layers.Dropout(rate=0.1)(ans_feature, training=True)
        cls_logits = self.dense_1_1(ans_feature)
        cls_logits = K.squeeze(cls_logits, -1)

        is_impossible = K.reshape(is_impossible, [-1])
        regression_loss = keras.layers.Lambda(lambda x:
                                              tf.nn.sigmoid_cross_entropy_with_logits(labels=x[0],
                                                                                      logits=x[1]))(
            [is_impossible, cls_logits])
        regression_loss = K.mean(regression_loss)
        total_loss += regression_loss * 0.5
        self.add_loss(total_loss, inputs=True)
        return total_loss

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'seq_len': self.max_seq_length,
        }
        base_config = super(XLnetLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
