from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import collections
# import json
# import math
# import os
# import modeling
# import optimization
# import tokenization
import numpy as np
import tensorflow as tf

def gen_batches(features, batch_size, num_epoch, drop_remainder=False, is_training=False):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_id = []

    for i in range(int(num_epoch)):
        for feature in features:
            #all_unique_ids.append(feature.unique_id)
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_id.append(feature.label_id)
            #all_start_positions.append(feature.start_position)
            #all_end_positions.append(feature.end_position)
            
            if len(all_input_ids) == batch_size:
                ids = np.copy(all_input_ids)
                mask = np.copy(all_input_mask)
                seg_ids = np.copy(all_segment_ids)
                labels = np.copy(all_label_id)
                all_input_ids = []
                all_input_mask = []
                all_segment_ids = []
                all_label_id = []
                yield ids,mask,seg_ids,np.array(labels).reshape((-1))

    if len(all_input_ids) > 0:
        yield all_input_ids, all_input_mask, all_segment_ids, all_label_id


def gen_batches_pretrain(features, batch_size, num_epoch, drop_remainder=False,
                         is_training=False):
    input_ids = []
    input_mask = []
    segment_ids = []
    masked_lm_positions = []
    masked_lm_ids = []
    masked_lm_weights = []
    next_sentence_labels = []
    
    for i in range(int(num_epoch)):
        for feature in features:
            input_ids.append(feature.input_ids)
            input_mask.append(feature.input_mask)
            segment_ids.append(feature.segment_ids)
            masked_lm_positions.append(feature.masked_lm_positions)
            masked_lm_ids.append(feature.masked_lm_ids)
            masked_lm_weights.append(feature.masked_lm_weights)
            next_sentence_labels.append(feature.next_sentence_labels)
            
            if len(input_ids) == batch_size:
                ids = np.copy(input_ids)
                mask = np.copy(input_mask)
                seg_ids = np.copy(segment_ids)
                pos = np.copy(masked_lm_positions)
                lm_ids = np.copy(masked_lm_ids)
                weights = np.copy(masked_lm_weights)
                labels = np.copy(next_sentence_labels)
                input_ids = []
                input_mask = []
                segment_ids = []
                masked_lm_positions = []
                masked_lm_ids = []
                masked_lm_weights = []
                next_sentence_labels = []
                yield ids, mask, seg_ids, pos, lm_ids, weights, labels

    if len(input_ids) > 0:
        yield input_ids,input_mask,segment_ids,masked_lm_positions,\
              masked_lm_ids,masked_lm_weights,next_sentence_labels


def gen_batches_squad(input_file, seq_length=128, batch_size=128,
                      num_train_epochs=1, is_training=True,
                      drop_remainder=False):
    feature = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }
    
    if is_training:
        feature["start_positions"] = tf.FixedLenFeature([], tf.int64)
        feature["end_positions"] = tf.FixedLenFeature([], tf.int64)
        
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(input_file, num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    
    _, value = reader.read_up_to(
        queue=filename_queue,
        num_records=batch_size)
    
    # _, serialized_example = reader.read(filename_queue)
    # # Decode the record read by the reader
    # features = tf.parse_single_example(serialized_example, features=feature)
    #
    # # # Convert the image data from string back to the numbers
    # # image = tf.decode_raw(features['train/image'], tf.float32)
    # # # Cast label data into int32
    # # label = tf.cast(features['train/label'], tf.int32)
    # # # Reshape image data into the original shape
    # # image = tf.reshape(image, [224, 224, 3])
    # # # Any preprocessing here ...
    #
    # # Creates batches by randomly shuffling tensors

    print('gen batch')
    # values = tf.train.batch(
    #     tensors=[value],
    #     batch_size=batch_size,
    #     num_threads=1,
    #     capacity=100 * batch_size,
    #     enqueue_many=True,
    #     allow_smaller_final_batch=False)

    values = tf.parse_example(value, features=feature)
    # values = tf.parse_example(values, features=feature)
    # features = tf.train.shuffle_batch(features,
    #                                 batch_size=10, capacity=30,
    #                                 num_threads=1, min_after_dequeue=10)
    
    return values
    

def gen_batches_squad2(input_file, seq_length=128, batch_size=128,
                      num_train_epochs=1, is_training=True,
                      drop_remainder=False):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    
    name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }
    
    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.float32)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.float32)
    
    def _decode_record(record, name_to_features=name_to_features):
        """Decodes a record to a TensorFlow example."""
        print('record', record)
        example = tf.parse_single_example(record, name_to_features)
        
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
          t = example[name]
          if t.dtype == tf.int64:
            t = tf.to_int32(t)
          example[name] = t
        
        return example
    
    """The actual input function."""
    # batch_size = params["batch_size"]
    
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    
    # Consider tf.contrib.data.parallel_interleave for parallelization
    dataset = d.interleave(tf.data.TFRecordDataset, cycle_length=num_train_epochs,
                           block_length=batch_size)
    # Consider passing num_parallel_calls or using tf.contrib.data.map_and_batch for performance
    dataset = dataset.map(_decode_record)
    # dataset = dataset.batch(num_train_epochs * batch_size)
    
    # d = d.apply(
    #     tf.contrib.data.map_and_batch(
    #         lambda record: _decode_record(record, name_to_features),
    #         batch_size=batch_size,
    #         drop_remainder=drop_remainder))
    
    return dataset
    # return dataset.make_one_shot_iterator().get_next()
  # def input_fn(params):
  #   """The actual input function."""
  #   batch_size = params["batch_size"]
  #
  #   # For training, we want a lot of parallel reading and shuffling.
  #   # For eval, we want no shuffling and parallel reading doesn't matter.
  #   d = tf.data.TFRecordDataset(input_file)
  #   if is_training:
  #     d = d.repeat()
  #     d = d.shuffle(buffer_size=100)
  #
  #   d = d.apply(
  #       tf.contrib.data.map_and_batch(
  #           lambda record: _decode_record(record, name_to_features),
  #           batch_size=batch_size,
  #           drop_remainder=drop_remainder))
  #
  #   return d
  #
  # return input_fn
