from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import modeling
import optimization
import tokenization
import six
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

# for running in jupyter env
flags.DEFINE_string('f', '', 'kernel')

# base_dir = '/mnt/scratch/chenqu/bert/uncased_L-12_H-768_A-12/'
base_dir = './uncased_L-12_H-768_A-12/'
output_dir = './data/output/'
data_dir = './data/MRPC/'

## Required parameters
flags.DEFINE_string(
    "bert_config_file", base_dir + "bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", base_dir + "vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", output_dir + "bert_coqa_out/1/",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", data_dir + "train.tsv",
                    "CoQA json for training. E.g., coqa-train-v1.0.json")

flags.DEFINE_string("predict_file", data_dir + "test.tsv",
    "CoQA json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", base_dir + "bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 12, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

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

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_integer(
    "history", 0,
    "Number of conversation history to use.")