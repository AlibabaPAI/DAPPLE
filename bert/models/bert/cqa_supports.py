from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import numpy as np
import tokenization
import six
import tensorflow as tf
from tensorflow import logging


class EvalResults(object):
    def __init__(self, capacity):
        self.metrics = {}
        self.capacity = capacity

    def add_dict(self, indict):
        for key,value in indict.iteritems():
            if key in self.metrics:
                if len(self.metrics[key]) == self.capacity:
                    self.metrics[key].pop(0)
            else:
                self.metrics[key] = []

            if isinstance(value, list):
                self.metrics[key].append(value[-1])
            else:
                self.metrics[key].append(value)

    def to_string(self):
        res = ["%s:%.2f"%(key, np.mean(self.metrics[key]))
               for key in self.metrics.keys()]
        return " ".join(res)


class CQAExample(object):
    """A single training/test example."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class InputPretrainExample(object):
    """A single training/test example for pretrain task."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, masked_lm_positions,
                 masked_lm_ids, masked_lm_weights, next_sentence_labels):

        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids
        self.masked_lm_weights = masked_lm_weights
        self.next_sentence_labels = next_sentence_labels


class InputCQAFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    import csv
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


def read_examples_do_nothing(input_file, is_training):
    """do nothing but just return input_file, reserved for tfrecord data"""
    return input_file

def read_textmatch_examples(input_file, is_training):
    """Creates examples for the training and dev sets."""
    if is_training:
        set_type = 'train'
    else:
        set_type = 'dev'

    examples = []
    for (i, line) in enumerate(read_tsv(input_file)):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text_a = tokenization.convert_to_unicode(line[3])
        text_b = tokenization.convert_to_unicode(line[4])
        label = tokenization.convert_to_unicode(line[0])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def read_cikm_examples(input_file, is_training):
    """Creates examples for the training and dev sets."""
    if is_training:
        set_type = 'train'
    else:
        set_type = 'dev'

    examples = []
    lengths = []
    for (i, line) in enumerate(read_tsv(input_file)):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        lengths.append(len(line[1].split()) + len(line[2].split()))
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = tokenization.convert_to_unicode(line[2])
        label = tokenization.convert_to_unicode(line[0])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    print('length', np.mean(lengths))
    raise Exception
    return examples


def read_review_examples(input_file, is_training):
    """Creates examples for the training and dev sets."""
    fold_id = 9  # fold 9 for training, the rest for testing
    if is_training:
        set_type = 'train'
    else:
        set_type = 'dev'

    examples = []
    lengths = []
    for (i, line) in enumerate(read_tsv(input_file)):
        # if is_training:
        #     if int(line[1]) == fold_id:
        #         continue
        # else:
        #     if int(line[1]) != fold_id:
        #         continue

        if int(line[1]) != fold_id:
            continue

        lengths.append(len(line[2].split()))
        # guid = "%s-%s" % (set_type, i)
        # text_a = tokenization.convert_to_unicode(line[2])
        # text_b = None
        # label = tokenization.convert_to_unicode(line[0])
        # examples.append(
        #     InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    print('length', np.mean(lengths))
    raise Exception
    return examples


def read_ae_examples(input_file, is_training):
    """Creates examples for the training and dev sets."""
    if is_training:
        set_type = 'train'
    else:
        set_type = 'dev'

    examples = []
    for (i, line) in enumerate(read_tsv(input_file)):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text_a = ' '.join(tokenization.convert_to_unicode(line[0]).split('|'))
        text_b = ' '.join(tokenization.convert_to_unicode(line[1]).split('|'))
        if float(line[2]) > 0.5:
            label = tokenization.convert_to_unicode('1')
        else:
            label = tokenization.convert_to_unicode('0')
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


def read_pretrain_examples(input_file, is_training):
    """Creates examples for the training and dev sets."""
    fold_id = 9  # fold 9 for training, the rest for testing
    if is_training:
        set_type = 'train'
    else:
        set_type = 'dev'

    examples = []
    for (i, line) in enumerate(read_tsv(input_file)):
        tokens = line
        if i < 3:
            print(i, line)
        if len(tokens) != 7:
            print(len(tokens))
            for (i, token) in enumerate(tokens):
                print(i, token)
            raise Exception

        guid = "%s-%s" % (set_type, i)
        # print(len(tokens[0].split(',')), len(tokens[1].split(',')),
        #       len(tokens[2].split(',')), len(tokens[3].split(',')),
        #       len(tokens[4].split(',')), len(tokens[5].split(',')),
        #       len(tokens[6].split(',')))
        examples.append(InputPretrainExample(
            guid=guid,
            input_ids=[int(idx) for idx in tokens[0].split(',')],
            input_mask=[int(idx) for idx in tokens[1].split(',')],
            segment_ids=[int(idx) for idx in tokens[2].split(',')],
            masked_lm_positions=[int(idx) for idx in tokens[3].split(',')],
            masked_lm_ids=[int(idx) for idx in tokens[4].split(',')],
            masked_lm_weights=[float(idx) for idx in tokens[5].split(',')],
            next_sentence_labels=int(tokens[6])))
    return examples


# def read_coqa_examples(input_file, is_training):
#     """Read a CoQA json file into a list of CQAExample."""
#     with tf.gfile.Open(input_file, "r") as reader:
#         input_data = json.load(reader)["data"]
#
#     def is_whitespace(c):
#         if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
#             return True
#         return False
#
#     examples = []
#     for entry in input_data[:10]:
#         paragraph_text = entry["story"]
#         doc_tokens = []
#         char_to_word_offset = []
#         prev_is_whitespace = True
#         for c in paragraph_text:
#             if is_whitespace(c):
#                 prev_is_whitespace = True
#             else:
#                 if prev_is_whitespace:
#                     doc_tokens.append(c)
#                 else:
#                     doc_tokens[-1] += c
#                 prev_is_whitespace = False
#             char_to_word_offset.append(len(doc_tokens) - 1)
#
#         ############################################################
#         # convert the convasational QAs to squad format, with history
#         ############################################################
#
#         story_id = entry['id']
#         questions = [(item['input_text'], story_id + str(item['turn_id'])) for item in entry['questions']] # [(question, question_id), ()]
#         answers = [(item['span_text'], item['span_start']) for item in entry['answers']]
#
#         qas = []
#         for i, (question, answer) in enumerate(zip(questions, answers)):
#             start_index = 0 if i - int(FLAGS.history) < 0 else i - int(FLAGS.history)
#             end_index = i
#             question_with_histories = ''
#             # prepend historical questions and answers
#             for each_question, each_answer in zip(questions[start_index: end_index], answers[start_index: end_index]):
#                 question_with_histories += each_question[0] + ' ' + each_answer[0] + ' '
#             # add the current question
#             question_with_histories += question[0]
#             if answer[1] == -1:
#                 qas.append({'id': question[1], 'question': question_with_histories, 'answers': [{'answer_start': -1, 'text': "unknown"}]})
#             else:
#                 qas.append({'id': question[1], 'question': question_with_histories, 'answers': [{'answer_start': answer[1], 'text': answer[0]}]})
#
#         for qa in qas:
#             qas_id = qa["id"]
#             question_text = qa["question"]
#             start_position = None
#             end_position = None
#             orig_answer_text = None
#
#             # if is_training:
#             # we read in the groundtruth answer bothing druing training and predicting, because we need to compute acc and f1 at predicting time.
#             if len(qa["answers"]) != 1:
#                 raise ValueError(
#                     "For training, each question should have exactly 1 answer.")
#             answer = qa["answers"][0]
#             orig_answer_text = answer["text"]
#             answer_offset = answer["answer_start"]
#             answer_length = len(orig_answer_text)
#             start_position = char_to_word_offset[answer_offset]
#             end_position = char_to_word_offset[answer_offset + answer_length - 1]
#             # Only add answers where the text can be exactly recovered from the
#             # document. If this CAN'T happen it's likely due to weird Unicode
#             # stuff so we will just skip the example.
#             #
#             # Note that this means for training mode, every example is NOT
#             # guaranteed to be preserved.
#             actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
#             cleaned_answer_text = " ".join(
#                 tokenization.whitespace_tokenize(orig_answer_text))
#             if actual_text.find(cleaned_answer_text) == -1:
#                 logging.warning("Could not find answer: '%s' vs. '%s'",
#                                    actual_text, cleaned_answer_text)
#                 continue
#
#             example = CQAExample(
#                 qas_id=qas_id,
#                 question_text=question_text,
#                 doc_tokens=doc_tokens,
#                 orig_answer_text=orig_answer_text,
#                 start_position=start_position,
#                 end_position=end_position)
#             examples.append(example)
#     return examples


def convert_examples_to_features_do_nothing(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """do nothing but just return examples, reserved for tfrecord data"""
    return examples


def convert_examples_to_features_cqa(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        # if is_training:
        # we do this for both training and predicting, because we need also start/end position at testing time to compute acc and f1
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if (example.start_position < doc_start or
                    example.end_position < doc_start or
                    example.start_position > doc_end or example.end_position > doc_end):
                    continue

                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
            else:
                # when predicting, we donot throw out any doc span to prevent label leaking
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

            if example_index < 20:
                logging.info("*** Example ***")
                logging.info("unique_id: %s" % (unique_id))
                logging.info("example_index: %s" % (example_index))
                logging.info("doc_span_index: %s" % (doc_span_index))
                logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logging.info("start_position: %d" % (start_position))
                    logging.info("end_position: %d" % (end_position))
                    logging.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position))
            unique_id += 1

    return features


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, model_type='classification'):
  """Loads a data file into a list of `InputBatch`s."""

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if model_type == 'classification':
        label_id = label_map[example.label]
    else:
        label_id = float(example.label)

    if ex_index < 5:
      logging.info("*** Example ***")
      logging.info("guid: %s" % (example.guid))
      logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      logging.info(
          "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      logging.info("label: %s (id = %d)" % (example.label, label_id))

    features.append(
        InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id))
  return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file):
    """Write final predictions to the json file."""
    logging.info("Writing predictions to: %s" % (output_prediction_file))
    logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_dict(train_batch):
    b_input_ids = train_batch['input_ids']
    b_input_mask = train_batch['input_mask']
    b_segment_ids = train_batch['segment_ids']
    b_labels = train_batch['label_id']

    return b_input_ids,b_input_mask,b_segment_ids,b_labels
