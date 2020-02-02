import numpy as np
from utils import backend as K
from utils import keras

from layers import LayerNormalization
from layers import FeedForward
from layers import RelativePartialMultiHeadSelfAttention


__all__ = [
    'get_custom_objects',
    'set_custom_objects',
    'build_transformer_xl',
]


def get_custom_objects():
    return {
        'AdaptiveEmbedding': AdaptiveEmbedding,
        'AdaptiveSoftmax': AdaptiveSoftmax,
        'Scale': Scale,
        'Memory': Memory,
        'LayerNormalization': LayerNormalization,
        'FeedForward': FeedForward,
        'PositionalEmbedding': PositionalEmbedding,
        'RelativeBias': RelativeBias,
        'RelativePartialMultiHeadSelfAttention': RelativePartialMultiHeadSelfAttention,
    }


def set_custom_objects():
    for name, layer in get_custom_objects().items():
        keras.utils.get_custom_objects()[name] = layer


def build_transformer_xl(units,
                         embed_dim,
                         hidden_dim,
                         num_token,
                         num_block,
                         num_head,
                         batch_size,
                         memory_len,
                         target_len,
                         dropout=0.0,
                         attention_dropout=0.0,
                         cutoffs=None,
                         div_val=1,
                         force_projection=None,
                         bind_embeddings=True,
                         bind_projections=True,
                         clamp_len=None,
                         share_biases=True):
    """Build transformer-XL model.

    :param units: Units inside the transformer.
    :param embed_dim: Dimension of embeddings.
    :param hidden_dim: Dimension inside position-wise feed-forward layer.
    :param num_token: Number of distinct input tokens.
    :param num_block: Number of basic encoder blocks.
    :param num_head: Number of heads for attention.
    :param batch_size: Maximum batch size.
    :param memory_len: The maximum length of memories.
    :param target_len: The length of prediction block.
    :param dropout: General dropout rate.
    :param attention_dropout: Dropout rate inside attention layer.
    :param cutoffs: Cutoffs of adaptive embedding.
    :param div_val: Scale factor of adaptive embedding.
    :param force_projection: Add projection when the dimensions are equal.
    :param bind_embeddings: Whether to bind embeddings to adaptive softmax.
    :param bind_projections: Whether to bind projections to adaptive softmax.
    :param clamp_len: The maximum value of relative position.
    :param share_biases: Whether to use the same biases for all layers.
    :return: The built model.
    """
    token_input = keras.layers.Input(shape=(target_len,), name='Input-Token')
    memory_length_input = keras.layers.Input(shape=(1,), name='Input-Memory-Length')
    inputs = [token_input, memory_length_input]

    results = AdaptiveEmbedding(
        input_dim=num_token,
        output_dim=units,
        embed_dim=embed_dim,
        cutoffs=cutoffs,
        div_val=div_val,
        mask_zero=True,
        force_projection=force_projection,
        return_embeddings=True,
        return_projections=True,
        name='Embed-Token',
    )(token_input)
    token_embed, embedding_weights = results[0], results[1:]
    token_embed = Scale(scale=np.sqrt(units), name='Embed-Token-Scaled')(token_embed)
    last_memory = Memory(
        batch_size=batch_size,
        memory_len=memory_len,
        target_len=target_len,
        output_dim=units,
        name='Memory-0',
    )([token_embed, memory_length_input])

    position_embed = PositionalEmbedding(
        output_dim=units,
        clamp_len=clamp_len,
        name='Embed-Position',
    )([token_input, last_memory])

    if 0.0 < dropout < 1.0:
        token_embed = keras.layers.Dropout(rate=dropout, name='Embed-Token-Dropped')(token_embed)
        position_embed = keras.layers.Dropout(rate=dropout, name='Embed-Position-Dropped')(position_embed)

    context_bias, relative_bias = None, None
    if share_biases:
        context_bias, relative_bias = RelativeBias(units=units, name='Biases')(last_memory)

    outputs = [token_embed]
    for i in range(num_block):
        block_input, block_output = outputs[-1], outputs[-1]
        if not share_biases:
            context_bias, relative_bias = RelativeBias(units=units, name='Biases-{}'.format(i + 1))(last_memory)
        block_output = RelativePartialMultiHeadSelfAttention(
            units=units,
            num_head=num_head,
            use_bias=False,
            attention_dropout=attention_dropout,
            name='Attention-{}'.format(i + 1),
        )([block_output, position_embed, last_memory, context_bias, relative_bias])
        block_output = keras.layers.Add(name='Attention-Res-{}'.format(i + 1))([block_input, block_output])
        if 0.0 < dropout < 1.0:
            block_output = keras.layers.Dropout(rate=dropout, name='Attention-Dropped-{}'.format(i + 1))(block_output)
        block_output = LayerNormalization(name='Attention-Norm-{}'.format(i + 1))(block_output)

        block_input = block_output
        block_output = FeedForward(
            units=hidden_dim,
            dropout_rate=dropout,
            name='FeedForward-{}'.format(i + 1),
        )(block_output)
        block_output = keras.layers.Add(name='FeedForward-Res-{}'.format(i + 1))([block_input, block_output])
        if 0.0 < dropout < 1.0:
            block_output = keras.layers.Dropout(rate=dropout, name='FeedForward-Dropped-{}'.format(i + 1))(block_output)
        block_output = LayerNormalization(name='FeedForward-Norm-{}'.format(i + 1))(block_output)

        if i < num_block - 1:
            last_memory = Memory(
                batch_size=batch_size,
                memory_len=memory_len,
                target_len=target_len,
                output_dim=units,
                name='Memory-{}'.format(i + 1),
            )([block_output, memory_length_input])

        outputs.append(block_output)

    softmax = AdaptiveSoftmax(
        input_dim=units,
        output_dim=num_token,
        embed_dim=embed_dim,
        cutoffs=cutoffs,
        div_val=div_val,
        force_projection=force_projection,
        bind_embeddings=bind_embeddings,
        bind_projections=bind_projections,
        name='Softmax',
    )(outputs[-1:] + embedding_weights)

    model = keras.models.Model(inputs=inputs, outputs=softmax)
    return model
