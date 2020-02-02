from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import collections
import json
import math
import os
import modeling
import optimization
import tokenization
import six
import tensorflow as tf

from cqa_supports import *
from cqa_flags import FLAGS
from cqa_model import *
from cqa_gen_batches import *

bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

tf.gfile.MakeDirs(FLAGS.output_dir)
tf.gfile.MakeDirs(FLAGS.output_dir + '/summaries/train/')
tf.gfile.MakeDirs(FLAGS.output_dir + '/summaries/val/')

tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

# read in training data, generate training features, and generate training batches
train_examples = None
num_train_steps = None
num_warmup_steps = None
train_file = FLAGS.train_file
train_examples = read_coqa_examples(input_file=train_file, is_training=True)
num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

# train_features = convert_examples_to_features(examples=train_examples, tokenizer=tokenizer, max_seq_length=FLAGS.max_seq_length,
#                                               doc_stride=FLAGS.doc_stride, max_query_length=FLAGS.max_query_length, is_training=True)
train_features = convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer)

# read in validation data, generate val features, and generate val batches
val_file = FLAGS.predict_file
val_examples = read_coqa_examples(input_file=val_file, is_training=False)
val_features = convert_examples_to_features(examples=val_examples, tokenizer=tokenizer, max_seq_length=FLAGS.max_seq_length, 
                                            doc_stride=FLAGS.doc_stride, max_query_length=FLAGS.max_query_length, is_training=False)
val_batches = cqa_gen_batches(val_features, FLAGS.predict_batch_size, 1)
num_val_examples = len(val_examples)


# tf Graph input
unique_ids = tf.placeholder(tf.int32, shape=[None], name='unique_ids')
input_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='input_ids')
input_mask = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='input_mask')
segment_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='segment_ids')
start_positions = tf.placeholder(tf.int32, shape=[None], name='start_positions')
end_positions = tf.placeholder(tf.int32, shape=[None], name='end_positions')
training = tf.placeholder(tf.bool, name='training')

is_training = FLAGS.is_training
(start_logits, end_logits) = cqa_model(
    bert_config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    use_one_hot_embeddings=True)

tvars = tf.trainable_variables()

initialized_variable_names = {}
if FLAGS.init_checkpoint:
    (assignment_map, initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
    tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

# tf.logging.info("**** Trainable Variables ****")
# for var in tvars:
#     init_string = ""
#     if var.name in initialized_variable_names:
#         init_string = ", *INIT_FROM_CKPT*"
#     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

# compute loss
seq_length = modeling.get_shape_list(input_ids)[1]
def compute_loss(logits, positions):
    one_hot_positions = tf.one_hot(
        positions, depth=seq_length, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss = -tf.reduce_mean(tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
    return loss

start_loss = compute_loss(start_logits, start_positions)
end_loss = compute_loss(end_logits, end_positions)
total_loss = (start_loss + end_loss) / 2.0

# get metrics
start_correct_prediction = tf.equal(tf.argmax(start_logits, -1), tf.argmax(start_positions, -1))
start_acc = tf.reduce_mean(tf.cast(start_correct_prediction, tf.float32))
end_correct_prediction = tf.equal(tf.argmax(end_logits, -1), tf.argmax(end_positions, -1))
end_acc = tf.reduce_mean(tf.cast(end_correct_prediction, tf.float32))

# for val
start_batch_correct_prediction = tf.reduce_sum(tf.cast(start_correct_prediction, tf.float32))
end_batch_correct_prediction = tf.reduce_sum(tf.cast(end_correct_prediction, tf.float32))

# write metrics to summary for tensorboard
tf.summary.scalar('start_loss', start_loss)
tf.summary.scalar('end_loss', end_loss)
tf.summary.scalar('total_loss', total_loss)
tf.summary.scalar('start_acc', start_acc)
tf.summary.scalar('end_acc', end_acc)
merged_summary_op = tf.summary.merge_all()

train_op = optimization.create_optimizer(total_loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, False)

tf.logging.info("***** Running training *****")
tf.logging.info("  Num orig examples = %d", len(train_examples))
tf.logging.info("  Num split examples = %d", len(train_features))
tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
tf.logging.info("  Num steps = %d", num_train_steps)

# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    if FLAGS.do_train:
        train_summary_writer = tf.summary.FileWriter(FLAGS.output_dir + 'summaries/train', sess.graph)
        val_summary_writer = tf.summary.FileWriter(FLAGS.output_dir + 'summaries/val')
        
        # Training cycle
        for step, train_batch in enumerate(train_batches):

            batch_unique_ids, batch_input_ids, batch_input_mask, batch_segment_ids, batch_start_positions, batch_end_positions = train_batch

            _, train_summary, start_acc_res, end_acc_res = sess.run([train_op, merged_summary_op, start_acc, end_acc], 
                                        feed_dict={unique_ids: batch_unique_ids, input_ids: batch_input_ids, 
                                          input_mask: batch_input_mask, segment_ids: batch_segment_ids, 
                                          start_positions: batch_start_positions, end_positions: batch_end_positions, training: False})
            train_summary_writer.add_summary(train_summary, step)
            tf.logging.info('training step: {0}, start_acc: {1}, end_acc: {2}'.format(step, start_acc_res, end_acc_res))
            
            if step % 100 == 0 and step != 0:
                total_val_start_correct_count, total_val_end_correct_count = 0, 0
                for step, train_batch in enumerate(train_batches):

                    batch_unique_ids, batch_input_ids, batch_input_mask, batch_segment_ids, \
                                                    batch_start_positions, batch_end_positions = train_batch

                    start_batch_correct_count, end_batch_correct_count = \
                                        sess.run([merged_summary_op, start_batch_correct_prediction, end_batch_correct_prediction], 
                                            feed_dict={unique_ids: batch_unique_ids, input_ids: batch_input_ids, 
                                            input_mask: batch_input_mask, segment_ids: batch_segment_ids, 
                                            start_positions: batch_start_positions, end_positions: batch_end_positions, training: False})
                    total_val_start_correct_count += start_batch_correct_count
                    total_val_end_correct_count += end_batch_correct_count
                
                val_start_acc = total_val_start_correct_count / num_val_examples
                val_end_acc = total_val_end_correct_count / num_val_examples
                val_summary = tf.Summary()
                val_summary.value.add(tag="start_acc", simple_value=val_start_acc)
                val_summary.value.add(tag="end_acc", simple_value=val_end_acc)
                val_summary_writer.add_summary(val_summary, step)
                
                tf.logging.info('\nevaluation: {0}, start_acc: {1}, end_acc: {2}\n'.format(step, start_acc_res, end_acc_res))
