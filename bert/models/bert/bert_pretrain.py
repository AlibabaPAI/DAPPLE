from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import tensorflow as tf
# import util

import optimization
from cqa_model import *
from cqa_supports import *
from cqa_gen_batches import gen_batches_pretrain
from tensorflow import logging
from eval_metrics import EvaluationMetrics

local = True
# candiate dataset: mrpc, cikm
dataset = "review"
model_types = ['classification', 'regression', 'pretrain']
bert_model_type = model_types[2]

base_dir = './bert_model_pretrain/'
init_checkpoint = None

# data reader
read_examples = read_pretrain_examples
gen_batches = gen_batches_pretrain

# train file
train_file = "./data/pretrain/sample_out.id.txt"
predict_file = "./data/pretrain/sample_out.id.txt"
output_dir = './data/output/'

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", output_dir,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 100,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# ## Required parameters
# flags.DEFINE_string(
#     "bert_config_file", base_dir + "bert_config.json",
#     "The config json file corresponding to the pre-trained BERT model. "
#     "This specifies the model architecture.")
#
# flags.DEFINE_string("vocab_file", base_dir + "vocab.txt",
#                     "The vocabulary file that the BERT model was trained on.")
#
# flags.DEFINE_string("output_dir", output_dir,
#                     "The output ckpt directory.")
#
## Other parameters
flags.DEFINE_string("train_file", train_file,
                    "CoQA json for training. E.g., coqa-train-v1.0.json")

flags.DEFINE_string("predict_file", predict_file,
                    "CoQA json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

# flags.DEFINE_string(
#     "init_checkpoint", init_checkpoint,
#     "Initial checkpoint (pre-trained model: base_dir + bert_model.ckpt).")
#
# flags.DEFINE_bool(
#     "do_lower_case", True,
#     "Whether to lower case the input text. Should be True for uncased "
#     "models and False for cased models.")
#
# flags.DEFINE_integer(
#     "max_seq_length", 64,
#     "The maximum total input sequence length after WordPiece tokenization. "
#     "Sequences longer than this will be truncated, and sequences shorter "
#     "than this will be padded.")
#
# flags.DEFINE_bool("do_train", True, "Whether to run training.")
#
# flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")
#
# flags.DEFINE_integer("train_batch_size", 56, "Total batch size for training.")
#
# flags.DEFINE_integer("predict_batch_size", 32,
#                      "Total batch size for predictions.")
#
# flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 20.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float("warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup for. "
                   "E.g., 0.1 = 10% of training.")

# flags.DEFINE_integer("save_checkpoints_steps", 1000,
#                      "How often to save the model checkpoint.")
#
# flags.DEFINE_integer("iterations_per_loop", 1000,
#                      "How many steps to make in each estimator call.")
#
# flags.DEFINE_integer("n_best_size", 20,
#                      "The total number of n-best predictions to generate in the "
#                      "nbest_predictions.json output file.")
#
# flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
#
# tf.flags.DEFINE_string("tpu_name", None,
#                        "The Cloud TPU to use for training. This should be either the name "
#                        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
#                        "url.")
#
# tf.flags.DEFINE_string("tpu_zone", None,
#                        "[Optional] GCE zone where the Cloud TPU is located in. If not "
#                        "specified, we will attempt to automatically detect the GCE project from "
#                        "metadata.")
#
# tf.flags.DEFINE_string("gcp_project", None,
#                        "[Optional] Project name for the Cloud TPU-enabled project. If not "
#                        "specified, we will attempt to automatically detect the GCE project from "
#                        "metadata.")
#
# tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
#
# flags.DEFINE_integer("num_tpu_cores", 8,
#                      "Only used if `use_tpu` is True. Total number of TPU cores to use.")
#
# flags.DEFINE_bool("verbose_logging", True,
#                   "If true, all of the warnings related to data processing will be printed. "
#                   "A number of warnings are expected for a normal SQuAD evaluation.")
#
# flags.DEFINE_integer("history", 0,
#                      "Number of conversation history to use.")


def main(_):
    # use tf.logging
    # logging = tf.logging
    tf.logging.set_verbosity(tf.logging.INFO)

    # get bert_config
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir + '/summaries/train/')
    tf.gfile.MakeDirs(FLAGS.output_dir + '/summaries/val/')

    # read in training data, generate training features, and generate training batches
    train_file = FLAGS.train_file
    train_examples = read_examples(input_file=train_file, is_training=True)
    num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    train_features = train_examples

    # tf Graph input
    input_ids = tf.placeholder(
        tf.int32, shape=[None, FLAGS.max_seq_length],
        name='input_ids')
    input_mask = tf.placeholder(
        tf.int32, shape=[None, FLAGS.max_seq_length],
        name='input_mask')
    segment_ids = tf.placeholder(
        tf.int32, shape=[None, FLAGS.max_seq_length],
        name='segment_ids')
    masked_lm_positions = tf.placeholder(
        tf.int32, shape=[None, FLAGS.max_predictions_per_seq],
        name='masked_lm_positions')
    masked_lm_ids = tf.placeholder(
        tf.int32, shape=[None, FLAGS.max_predictions_per_seq],
        name='masked_lm_ids')
    masked_lm_weights = tf.placeholder(
        tf.float32, shape=[None, FLAGS.max_predictions_per_seq],
        name='masked_lm_weights')
    next_sentence_labels = tf.placeholder(
        tf.int32, shape=[None],
        name='next_sentence_labels')
    training = tf.placeholder(tf.bool, name='training')
    kwargs = dict()
    kwargs['masked_lm_positions'] = masked_lm_positions
    kwargs['masked_lm_ids'] = masked_lm_ids
    kwargs['masked_lm_weights'] = masked_lm_weights
    kwargs['next_sentence_labels'] = next_sentence_labels

    model = BertFinetune(bert_config=bert_config,
                         is_training=True,
                         input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         labels=None,
                         num_labels=None,
                         use_one_hot_embeddings=False,
                         model_type=bert_model_type,
                         kwargs=kwargs)
    model_name = 'bert-L%d-H%d-A%d' % (bert_config.num_hidden_layers,
                                       bert_config.hidden_size,
                                       bert_config.num_attention_heads)

    tvars = tf.trainable_variables()

    if FLAGS.init_checkpoint:
        (assignment_map, initialized_variable_names) = \
            modeling.get_assigment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            else:
                # un_init_variables.add(var)
                pass
            logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
    else:
        print('no init from checkpoint')

    logging.info("build summary and optimizer")
    model.total_loss = model.loss
    # write metrics to summary for tensorboard
    tf.summary.scalar('total_loss', model.total_loss)
    model.merged_summary_op = tf.summary.merge_all()
    model.train_op,model.inc_step = optimization.create_optimizer(
        model.total_loss, FLAGS.learning_rate,
        num_train_steps, num_warmup_steps, False)

    logging.info("***** Running training *****")
    logging.info("  Num orig examples = %d", len(train_examples))
    logging.info("  Num split examples = %d", len(train_features))
    logging.info("  Batch size = %d", FLAGS.train_batch_size)
    logging.info("  Num steps = %d", num_train_steps)

    # Initializing the variables
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    eval_results = EvalResults(capacity=2)
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    session_config.log_device_placement = False
    with tf.Session(config=session_config) as sess:
        sess.run(init)
        sess.run(local_init)
        # restore
        model.restore(output_dir, sess)

        if FLAGS.do_train:
            train_summary_writer = tf.summary.FileWriter(FLAGS.output_dir + 'summaries/train', sess.graph)

            # Training cycle
            step = 0
            for ids, mask, seg_ids, pos, lm_ids, weights, labels in gen_batches(
                    train_features, FLAGS.train_batch_size, FLAGS.num_train_epochs):
                if step == 0:
                    tf.logging.info('\tids {}'.format(ids[0, :]))
                    tf.logging.info('\tmask {}'.format(mask[0, :]))
                    tf.logging.info('\tseg_ids {}'.format(seg_ids[0, :]))
                    tf.logging.info('\tpos {}'.format(pos[0, :]))
                    tf.logging.info('\tlm_ids {}'.format(lm_ids[0, :]))
                    tf.logging.info('\tweights {}'.format(weights[0, :]))
                    tf.logging.info('\tlabels {}'.format(labels))
                step += 1
                _, _, train_summary, r_loss, evalm = sess.run(
                    [model.train_op, model.inc_step, model.merged_summary_op, model.total_loss,
                     model.eval_metric],
                    feed_dict={input_ids: ids,
                               input_mask: mask,
                               segment_ids: seg_ids,
                               masked_lm_positions: pos,
                               masked_lm_ids: lm_ids,
                               masked_lm_weights: weights,
                               next_sentence_labels: labels,
                               training: True})

                eval_results.add_dict(evalm)
                train_summary_writer.add_summary(train_summary, step)
                if step % 10 == 0:
                    logging.info('training step: {0}, loss: {1}, {2}'.format(
                        step, r_loss, eval_results.to_string()))
                    model.save(output_dir + model_name, sess, step)


if __name__ == "__main__":
  tf.app.run()
