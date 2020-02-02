from utils import backend as K
from utils import keras


class RelativeBias(keras.layers.Layer):

    def __init__(self,
                 num_layer,
                 num_head,
                 attention_head_dim,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(RelativeBias, self).__init__(**kwargs)
        self.num_layer = num_layer
        self.num_head = num_head
        self.attention_head_dim = attention_head_dim
        self.supports_masking = True
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.bias_context, self.bias_relative, self.bias_segment = None, None, None

    def compute_output_shape(self, input_shape):
        return [(self.num_layer, self.num_head, self.attention_head_dim)] * 3

    def build(self, input_shape):
        self.bias_context = self.add_weight(
            shape=(self.num_layer, self.num_head, self.attention_head_dim),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=K.floatx(),
            name='bias_context',
        )
        self.bias_relative = self.add_weight(
            shape=(self.num_layer, self.num_head, self.attention_head_dim),
            #shape=(self.num_head, self.attention_head_dim),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=K.floatx(),
            name='bias_relative',
        )

        self.bias_segment = self.add_weight(
            shape=(self.num_layer, self.num_head, self.attention_head_dim),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=K.floatx(),
            name='bias_segment',
        )
        super(RelativeBias, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return [
            K.identity(self.bias_context),
            K.identity(self.bias_relative),
            K.identity(self.bias_segment),
        ]

    def get_config(self):
        config = {
            'num_layer': self.num_layer,
            'num_head': self.num_head,
            'attention_head_dim': self.attention_head_dim,
        }
        base_config = super(RelativeBias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
