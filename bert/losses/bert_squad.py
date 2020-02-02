import tensorflow as tf

from losses.loss_base import BaseLoss
from models.bert.bert import BertFinetune, BertFinetuneSlice
from train_flags import FLAGS


class bert_squad(BaseLoss):

  def loss_fn(self):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      with tf.device('/cpu:0'):
        next_batch = self._dataset_iterator.get_next()
      kwargs = dict()
      kwargs['start_positions'] = next_batch['start_positions']
      kwargs['end_positions'] = next_batch['end_positions']
      kwargs['unique_ids'] = next_batch['unique_ids']
      import os
      bert_config_file = os.path.join(FLAGS.model_dir, FLAGS.model_config_file_name)
      model = BertFinetune(bert_config_file=bert_config_file,
                           max_seq_length=FLAGS.max_seq_length,
                           is_training=True,
                           input_ids=next_batch['input_ids'],
                           input_mask=next_batch['input_mask'],
                           segment_ids=next_batch['segment_ids'],
                           labels=None,
                           use_one_hot_embeddings=False,
                           model_type=FLAGS.model_type,
                           kwargs=kwargs)
      self._loss = model.loss

    return self._loss

  def stages(self, slice_devices=["/device:GPU:0"], dep_loss=None, dep_outputs=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      with tf.device('/cpu:0'):
        next_batch = self._dataset_iterator.get_next()
      kwargs = dict()
      kwargs['start_positions'] = next_batch['start_positions']
      kwargs['end_positions'] = next_batch['end_positions']
      kwargs['unique_ids'] = next_batch['unique_ids']
      import os
      bert_config_file = os.path.join(FLAGS.model_dir, FLAGS.model_config_file_name)
      model = BertFinetuneSlice(bert_config_file=bert_config_file,
                           max_seq_length=FLAGS.max_seq_length,
                           is_training=True,
                           input_ids=next_batch['input_ids'],
                           input_mask=next_batch['input_mask'],
                           segment_ids=next_batch['segment_ids'],
                           labels=None,
                           use_one_hot_embeddings=False,
                           model_type=FLAGS.model_type,
                           slice_devices=slice_devices,
                           dep_outputs=dep_outputs,
                           kwargs=kwargs)

      self._stage_outputs = model.stage_outputs
      self._loss = model.loss
      print (self._loss, self._stage_outputs[-1])
    return self._loss, self._stage_outputs
