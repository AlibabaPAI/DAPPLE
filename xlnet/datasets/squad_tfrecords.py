import tensorflow as tf
from tensorflow.contrib.distribute.python import prefetching_ops_v2

def input_fn_builder(input_glob, batch_size, is_training, seq_length=512):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "cls_index": tf.FixedLenFeature([], tf.int64),
        "p_mask": tf.FixedLenFeature([seq_length], tf.float32)
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["is_impossible"] = tf.FixedLenFeature([], tf.float32)
    else:
        name_to_features["unique_ids"] = tf.FixedLenFeature([], tf.int64)

    def input_fn():
        d = tf.data.TFRecordDataset(input_glob)
        if is_training:
            d = d.shuffle(buffer_size=1024)
            d = d.repeat()
        else:
            d = d.repeat(1)

        def _decode_record(record, name_to_features):
            example = tf.parse_single_example(record, name_to_features)
            return example

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=8,
                drop_remainder=False))
        d = d.prefetch(1024)
        if False:
            d = d.apply(tf.contrib.data.prefetch_to_device('/replica:0/task:0/device:GPU:0'))
        return d

    return input_fn
