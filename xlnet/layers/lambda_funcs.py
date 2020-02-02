import tensorflow as tf

from utils import backend as K
from utils import keras


def get_attn_mask_func(inputs):
    input_mask = inputs
    batch_size = K.shape(input_mask)[0]
    target_len = K.shape(input_mask)[1]
    # 512, ?
    input_mask_trans = K.transpose(input_mask)

    # 1, 512, ?
    # data_mask = input_mask[None]
    data_mask = K.expand_dims(input_mask_trans, axis=0)

    # ?, 0, 2
    # mems_mask = tf.zeros([tf.shape(data_mask)[0], mlen, bsz], dtype=tf_float)
    mems_mask = keras.layers.Lambda(lambda x: tf.zeros([1, 0, x], dtype=tf.float32))(batch_size)

    # 1, 512, 2
    # data_mask = tf.concat([mems_mask, data_mask], 1)
    data_mask = K.concatenate([mems_mask, data_mask], axis=1)

    # attn_mask = data_mask[:, :, :, None]
    attn_mask = K.expand_dims(data_mask, axis=-1)

    # attn_mask = tf.cast(attn_mask > 0, dtype=tf_float)
    attn_mask = keras.layers.Lambda(lambda x: tf.cast(x > 0, dtype=tf.float32))(attn_mask)

    # non_tgt_mask = -tf.eye(512, dtype=tf.float32)
    non_tgt_mask = keras.layers.Lambda(lambda x: -tf.eye(x, dtype=tf.float32))(target_len)

    # non_tgt_mask = tf.concat([tf.zeros([qlen, mlen], dtype=tf_float),non_tgt_mask], axis=-1)
    tmp = keras.layers.Lambda(lambda x: tf.zeros([x, 0], dtype=tf.float32))(target_len)
    non_tgt_mask = K.concatenate([tmp, non_tgt_mask], axis=-1)

    # non_tgt_mask = tf.cast((attn_mask + non_tgt_mask[:, :, None, None]) > 0, dtype=tf_float)
    non_tgt_mask = K.expand_dims(non_tgt_mask, axis=-1)
    non_tgt_mask = K.expand_dims(non_tgt_mask, axis=-1)
    tmp2 = K.greater(attn_mask + non_tgt_mask, 0)
    attn_mask = K.cast(tmp2, tf.float32)
    return attn_mask


get_attn_mask = keras.layers.Lambda(lambda x: get_attn_mask_func(x), name="ATTN_MASK")


def get_pos_emb_func(inputs):
    input_ids, token_embed = inputs

    batch_size = K.shape(input_ids)[1]
    target_len = K.shape(input_ids)[0]
    embedding_dim = K.shape(token_embed)[2]

    # freq_seq = tf.range(0, d_model, 2.0)
    freq_seq = keras.layers.Lambda(lambda x: tf.range(0, x, 2.0))(embedding_dim)
    inv_freq = 1 / (10000 ** (freq_seq / K.cast(embedding_dim, tf.float32)))

    beg, end = target_len, -target_len

    # fwd_pos_seq = tf.range(beg, end, -1.0)
    fwd_pos_seq = keras.layers.Lambda(lambda x: tf.range(x[0], x[1], -1.0))([beg, end])

    # sinusoid_inp = tf.einsum('i,d->id', pos_seq, inv_freq)
    sinusoid_inp = keras.layers.Lambda(lambda x: tf.einsum('i,d->id', x[0], x[1]))([fwd_pos_seq, inv_freq])
    sinusoid_inp = K.cast(sinusoid_inp, tf.float32)

    # pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    pos_emb = keras.layers.Lambda(lambda x: tf.concat([tf.sin(x), tf.cos(x)], -1))(sinusoid_inp)

    # pos_emb = pos_emb[:, None, :]
    pos_emb = K.expand_dims(pos_emb, axis=1)

    # pos_emb = tf.tile(pos_emb, [1, bsz, 1])
    pos_emb = K.tile(pos_emb, [1, batch_size, 1])

    return pos_emb


get_pos_emb = keras.layers.Lambda(lambda x: get_pos_emb_func(x), name="POS-EMB")


def rel_shift_func(x, klen):
    """perform relative shift to form the relative attention score."""
    x_size = tf.shape(x)
    x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
    x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

    return x


rel_shift = keras.layers.Lambda(lambda x: rel_shift_func(x[0], x[1]), name="Rel-Shift")
