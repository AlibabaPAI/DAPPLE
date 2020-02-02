# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import optimization
import tensorflow as tf
import tokenization
import util

import modeling
from cqa_model import BertFinetune
from cqa_gen_batches import gen_batches
from tensorflow import logging
from eval_metrics import EvaluationMetrics

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

local = True
datasets = ['mrpc', "cikm", 'review', 'ae']
dataset = datasets[0]
model_types = ['classification', 'regression']
bert_model_type = model_types[0]

if local:
    # base_dir = './bert_model_toy/'
    init_checkpoint = None
    base_dir = './bert_model_pretrain/'
    # init_checkpoint = 'bert_model_pretrain/ae_ru/bert-L3-H128-A4-20'
    # output_dir = './data/output/'
else:
    base_dir = './uncased_L-12_H-768_A-12/'
    init_checkpoint = base_dir + 'bert_model.ckpt'

# data reader
if dataset == 'mrpc':
    from cqa_supports import read_textmatch_examples as read_examples
    from cqa_supports import convert_examples_to_features
    train_file = "./data/mrpc_bert/train.tsv"
    predict_file = "./data/mrpc_bert/test.tsv"
    output_dir = './data/output/'
elif dataset == 'cikm':
    from cqa_supports import read_cikm_examples as read_examples
    train_file = "./data/cikm.data/test.txt"
    predict_file = "./data/cikm.data/test.txt"
    output_dir = './data/output/'
elif dataset == 'review':
    from cqa_supports import read_review_examples as read_examples
    train_file = "./data/review/electronics.toy.txt"
    predict_file = "./data/review/electronics.toy.txt"
    output_dir = './data/output/'
elif dataset == 'ae':
    from cqa_supports import read_ae_examples as read_examples
    train_file = "../../TransferLearning/ATP/examples/AE/data_ru/" \
                 "RU_evaluate_with_candidate_1019_seg.csv"
    predict_file = "../../TransferLearning/ATP/examples/AE/data_ru/" \
                   "RU_evaluate_with_candidate_1019_seg.csv"
    output_dir = './data/output/'


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", base_dir + "bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", base_dir + "vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", output_dir + "textmatch/",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", train_file,
                    "CoQA json for training. E.g., coqa-train-v1.0.json")

flags.DEFINE_string("predict_file", predict_file,
    "CoQA json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", init_checkpoint,
    "Initial checkpoint (pre-trained model: base_dir + bert_model.ckpt).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 64,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 56, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 32,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 20.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float("warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup for. "
                   "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("n_best_size", 20,
                     "The total number of n-best predictions to generate in the "
                     "nbest_predictions.json output file.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string("tpu_name", None,
                       "The Cloud TPU to use for training. This should be either the name "
                       "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
                       "url.")

tf.flags.DEFINE_string("tpu_zone", None,
                       "[Optional] GCE zone where the Cloud TPU is located in. If not "
                       "specified, we will attempt to automatically detect the GCE project from "
                       "metadata.")

tf.flags.DEFINE_string("gcp_project", None,
                       "[Optional] Project name for the Cloud TPU-enabled project. If not "
                       "specified, we will attempt to automatically detect the GCE project from "
                       "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer("num_tpu_cores", 8,
                     "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("verbose_logging", True,
                  "If true, all of the warnings related to data processing will be printed. "
                  "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_integer("history", 0,
                     "Number of conversation history to use.")


def main(_):
    # use tf.logging
    # logging = tf.logging
    tf.logging.set_verbosity(tf.logging.INFO)

    # get bert_config
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    label_list = ['0', '1']
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir + '/summaries/train/')
    tf.gfile.MakeDirs(FLAGS.output_dir + '/summaries/val/')

    # tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file,
    #                                        do_lower_case=FLAGS.do_lower_case)
    tokenizer = tokenization.BasicTokenizerWrapper(vocab_file=FLAGS.vocab_file,
                                                   do_lower_case=FLAGS.do_lower_case)

    # read in training data, generate training features, and generate training batches
    train_file = FLAGS.train_file
    train_examples = read_examples(input_file=train_file, is_training=True)
    if len(train_examples) > 10:
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    else:
        num_train_steps = 100000 * FLAGS.num_train_epochs
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    train_features = convert_examples_to_features(
        train_examples, label_list=label_list,
        max_seq_length=FLAGS.max_seq_length, tokenizer=tokenizer,
        model_type=bert_model_type)

    # read in validation data, generate val features, and generate val batches
    val_file = FLAGS.predict_file
    val_examples = read_examples(input_file=val_file, is_training=False)
    val_features = convert_examples_to_features(
        val_examples, label_list=label_list,
        max_seq_length=FLAGS.max_seq_length, tokenizer=tokenizer,
        model_type=bert_model_type)

    # tf Graph input
    input_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length], name='segment_ids')
    if bert_model_type == 'classification':
        label_id = tf.placeholder(tf.int32, shape=[None], name='labels')
    else:
        label_id = tf.placeholder(tf.float32, shape=[None], name='labels')
    training = tf.placeholder(tf.bool, name='training')

    model = BertFinetune(bert_config=bert_config,
                         is_training=True,
                         input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         labels=label_id,
                         num_labels=2,
                         use_one_hot_embeddings=False,
                         model_type=bert_model_type)
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
    model.train_op,model.inc_step = optimization.create_optimizer(model.total_loss, FLAGS.learning_rate,
                                                   num_train_steps, num_warmup_steps,
                                                   False)

    logging.info("***** Running training *****")
    logging.info("  Num orig examples = %d", len(train_examples))
    logging.info("  Num split examples = %d", len(train_features))
    logging.info("  Num valid examples = %d", len(val_examples))
    logging.info("  Num valid split examples = %d", len(val_features))
    logging.info("  Batch size = %d", FLAGS.train_batch_size)
    logging.info("  Num steps = %d", num_train_steps)

    # evaluation metrics
    if bert_model_type == 'classification':
        metrics = EvaluationMetrics(eval_conf=['auc', 'accuracy'])
    else:
        metrics = EvaluationMetrics(eval_conf=['corr', 'auc', 'mse'])

    # Initializing the variables
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    # with tf.Session() as sess:
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    session_config.log_device_placement = False
    with tf.Session(config=session_config) as sess:
        sess.run(init)
        sess.run(local_init)

        if FLAGS.do_train:
            train_summary_writer = tf.summary.FileWriter(FLAGS.output_dir + 'summaries/train', sess.graph)
            val_summary_writer = tf.summary.FileWriter(FLAGS.output_dir + 'summaries/val')

            # Training cycle
            step = 0
            for ids,mask,seg_ids,labels in gen_batches(train_features,
                                                       FLAGS.predict_batch_size,
                                                       FLAGS.num_train_epochs):
                step += 1
                _, _, train_summary, r_loss = sess.run(
                    [model.train_op, model.inc_step, model.merged_summary_op, model.total_loss],
                    feed_dict={input_ids: ids,
                               input_mask: mask,
                               segment_ids: seg_ids,
                               label_id: labels,
                               training: True})
                train_summary_writer.add_summary(train_summary, step)
                if step % 10 == 0:
                    logging.info('training step: {0}, loss: {1}'.format(step, r_loss))

                if step % 50 == 0:
                    # save model
                    model.save(output_dir + model_name, sess, step)

                    results = [[], []]
                    val_acc,val_loss = 0.0, 0.0
                    v_step = 0
                    r_losses = []
                    for ids,mask,seg_ids,labels in gen_batches(val_features,
                                                               FLAGS.predict_batch_size,
                                                               1):
                        test_summary, r_loss, r_logits = sess.run(
                            [model.merged_summary_op, model.total_loss, model.logits],
                            feed_dict={input_ids: ids,
                                       input_mask: mask,
                                       segment_ids: seg_ids,
                                       label_id: labels,
                                       training: False})

                        if bert_model_type == 'classification':
                            predictions = util.softmax(r_logits)[:, -1]
                        else:
                            predictions = r_logits

                        results[0].extend(np.array(predictions).reshape((-1)))
                        results[1].extend(np.array(labels).reshape((-1)))
                        r_losses.append(r_loss)
                        if v_step % 100 == 0:
                            logging.info('logits shape {}, r_loss {}'.format(np.array(r_logits).shape,
                                                                             np.mean(r_losses)))
                        v_step += 1

                    util.evaluate(results, metrics, interval=None)

                    val_summary = tf.Summary()
                    val_summary.value.add(tag="loss", simple_value=val_loss)
                    val_summary_writer.add_summary(val_summary, step)

                    logging.info('evaluation: {0}\n'.format(step))


if __name__ == "__main__":
  tf.app.run()
