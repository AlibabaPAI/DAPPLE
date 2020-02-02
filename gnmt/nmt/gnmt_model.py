# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""GNMT attention sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from . import attention_model
from . import model_helper
from .common import cluster_utils
from .utils import misc_utils as utils
from .utils import vocab_utils

import pdb

__all__ = ["GNMTModel"]

class GNMTModel(attention_model.AttentionModel):
  """Sequence-to-sequence dynamic model with GNMT attention architecture.
  """

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):
    self.is_gnmt_attention = (
        hparams.attention_architecture in ["gnmt", "gnmt_v2"])

    super(GNMTModel, self).__init__(
        hparams=hparams,
        mode=mode,
        iterator=iterator,
        source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope,
        extra_args=extra_args)

  def _build_encoder(self, hparams):
    """Build a GNMT encoder."""
    if hparams.encoder_type == "uni" or hparams.encoder_type == "bi":
      return super(GNMTModel, self)._build_encoder(hparams)

    if hparams.encoder_type != "gnmt":
      raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

    # Build GNMT encoder.
    num_bi_layers = 1
    num_uni_layers = self.num_encoder_layers - num_bi_layers
    utils.print_out("# Build a GNMT encoder")
    utils.print_out("  num_bi_layers = %d" % num_bi_layers)
    utils.print_out("  num_uni_layers = %d" % num_uni_layers)

    iterator = self.iterator
    source = iterator.source
    if self.time_major:
      source = tf.transpose(source)

    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE) as scope:
      dtype = scope.dtype

      ## Dapple guard
      DAPPLE_TEST=hparams.dapple_test

      devices = cluster_utils.get_pipeline_devices(hparams.pipeline_device_num)
      encoder_device=devices[0]

      self.encoder_emb_inp = self.encoder_emb_lookup_fn(
          self.embedding_encoder, source)


      # Execute _build_bidirectional_rnn from Model class
      if DAPPLE_TEST:
        fw_cell = model_helper._single_cell_dapple(hparams, self.mode, 0,
                                                   residual_connection=False, device_str=encoder_device)
        bw_cell = model_helper._single_cell_dapple(hparams, self.mode, 0,
                                                   residual_connection=False, device_str=encoder_device)
        bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            self.encoder_emb_inp,
            dtype=dtype,
            sequence_length=iterator.source_sequence_length,
            time_major=self.time_major,
            swap_memory=True)
        bi_encoder_outputs = tf.concat(bi_outputs, -1)
      else:
        bi_encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
            inputs=self.encoder_emb_inp,
            sequence_length=iterator.source_sequence_length,
            dtype=dtype,
            hparams=hparams,
            num_bi_layers=num_bi_layers,
            num_bi_residual_layers=0,  # no residual connection
            base_gpu=0,
        )


      ## Build unidirectional layers
      if self.extract_encoder_layers:
        encoder_state, encoder_outputs = self._build_individual_encoder_layers(
            bi_encoder_outputs, num_uni_layers, dtype, hparams)
      elif DAPPLE_TEST:
        ## All Encoder layers' state list
        self.encoder_state_list = [bi_encoder_outputs[:, :, :hparams.num_units],
                                   bi_encoder_outputs[:, :, hparams.num_units:]]
        encoder_state = []
        with tf.variable_scope("rnn/multi_rnn_cell"):
          # Encoder layer1
          uni_cell1 = model_helper._single_cell_dapple(hparams, self.mode, 1,
                                                       residual_connection=False, device_str=encoder_device)
          uc1_outs, uc1_stat = self._dynamic_rnn_helper(uni_cell1, bi_encoder_outputs, 1, dtype)
          self.encoder_state_list.append(uc1_outs)
          encoder_state.append(uc1_stat)
          # Encoder layer2 (from this layer *residual connection*=True)
          uc2 = model_helper._single_cell_dapple(hparams, self.mode, 2, True, encoder_device)
          uc2_outs, uc2_stat = self._dynamic_rnn_helper(uc2, uc1_outs, 2, dtype)
          self.encoder_state_list.append(uc2_outs)
          encoder_state.append(uc2_stat)
          # Encoder layer3
          uc3 = model_helper._single_cell_dapple(hparams, self.mode, 3, True, encoder_device)
          uc3_outs, uc3_stat = self._dynamic_rnn_helper(uc3, uc2_outs, 3, dtype)
          self.encoder_state_list.append(uc3_outs)
          encoder_state.append(uc3_stat)
          # Encoder layer4
          uc4 = model_helper._single_cell_dapple(hparams, self.mode, 4, True, encoder_device)
          uc4_outs, uc4_stat = self._dynamic_rnn_helper(uc4, uc3_outs, 4, dtype)
          self.encoder_state_list.append(uc4_outs)
          encoder_state.append(uc4_stat)
          # Encoder layer5
          uc5 = model_helper._single_cell_dapple(hparams, self.mode, 5, True, encoder_device)
          uc5_outs, uc5_stat = self._dynamic_rnn_helper(uc5, uc4_outs, 5, dtype)
          self.encoder_state_list.append(uc5_outs)
          encoder_state.append(uc5_stat)
          # Encoder layer6
          uc6 = model_helper._single_cell_dapple(hparams, self.mode, 6, True, encoder_device)
          uc6_outs, uc6_stat = self._dynamic_rnn_helper(uc6, uc5_outs, 6, dtype)
          self.encoder_state_list.append(uc6_outs)
          encoder_state.append(uc6_stat)
          # Encoder layer7
          uc7 = model_helper._single_cell_dapple(hparams, self.mode, 7, True, encoder_device)
          uc7_outs, uc7_stat = self._dynamic_rnn_helper(uc7, uc6_outs, 7, dtype)
          self.encoder_state_list.append(uc7_outs)
          encoder_state.append(uc7_stat)
          if hparams.gnmt16:
            # Encoder layer8
            uc8 = model_helper._single_cell_dapple(hparams, self.mode, 8, True, encoder_device)
            uc8_outs, uc8_stat = self._dynamic_rnn_helper(uc8, uc7_outs, 8, dtype)
            self.encoder_state_list.append(uc8_outs)
            encoder_state.append(uc8_stat)
            # Encoder layer9
            uc9 = model_helper._single_cell_dapple(hparams, self.mode, 9, True, encoder_device)
            uc9_outs, uc9_stat = self._dynamic_rnn_helper(uc9, uc8_outs, 9, dtype)
            self.encoder_state_list.append(uc9_outs)
            encoder_state.append(uc9_stat)
            # Encoder layer10
            uc10 = model_helper._single_cell_dapple(hparams, self.mode, 10, True, encoder_device)
            uc10_outs, uc10_stat = self._dynamic_rnn_helper(uc10, uc9_outs, 10, dtype)
            self.encoder_state_list.append(uc10_outs)
            encoder_state.append(uc10_stat)
            # Encoder layer11
            uc11 = model_helper._single_cell_dapple(hparams, self.mode, 11, True, encoder_device)
            uc11_outs, uc11_stat = self._dynamic_rnn_helper(uc11, uc10_outs, 11, dtype)
            self.encoder_state_list.append(uc11_outs)
            encoder_state.append(uc11_stat)
            # Encoder layer12
            uc12 = model_helper._single_cell_dapple(hparams, self.mode, 12, True, encoder_device)
            uc12_outs, uc12_stat = self._dynamic_rnn_helper(uc12, uc11_outs, 12, dtype)
            self.encoder_state_list.append(uc12_outs)
            encoder_state.append(uc12_stat)
            # Encoder layer13
            uc13 = model_helper._single_cell_dapple(hparams, self.mode, 13, True, encoder_device)
            uc13_outs, uc13_stat = self._dynamic_rnn_helper(uc13, uc12_outs, 13, dtype)
            self.encoder_state_list.append(uc13_outs)
            encoder_state.append(uc13_stat)
            # Encoder layer14
            uc14 = model_helper._single_cell_dapple(hparams, self.mode, 14, True, encoder_device)
            uc14_outs, uc14_stat = self._dynamic_rnn_helper(uc14, uc13_outs, 14, dtype)
            self.encoder_state_list.append(uc14_outs)
            encoder_state.append(uc14_stat)
            # Encoder layer15
            uc15 = model_helper._single_cell_dapple(hparams, self.mode, 15, True, encoder_device)
            uc15_outs, uc15_stat = self._dynamic_rnn_helper(uc15, uc14_outs, 15, dtype)
            self.encoder_state_list.append(uc15_outs)
            encoder_state.append(uc15_stat)

        ## Encoder last layer output
        encoder_outputs = uc7_outs if not hparams.gnmt16 else uc15_outs
        encoder_state = tuple(encoder_state)
      else:
        encoder_state, encoder_outputs = self._build_all_encoder_layers(
            bi_encoder_outputs, num_uni_layers, dtype, hparams)

      # Pass all encoder states to the decoder
      #   except the first bi-directional layer
      encoder_state = (bi_encoder_state[1],) + (
          (encoder_state,) if num_uni_layers == 1 else encoder_state)

    return encoder_outputs, encoder_state

  def _dynamic_rnn_helper(self, cell, inputs, layer_id, dtype):
    with tf.variable_scope("cell_%d" % layer_id) as scope:
      return tf.nn.dynamic_rnn(cell, inputs,
                               dtype=dtype, sequence_length=self.iterator.source_sequence_length,
                               time_major=self.time_major)

  def _build_all_encoder_layers(self, bi_encoder_outputs,
                                num_uni_layers, dtype, hparams):
    """Build encoder layers all at once."""
    uni_cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=num_uni_layers,
        num_residual_layers=self.num_encoder_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=self.num_gpus,
        base_gpu=0,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        uni_cell,
        bi_encoder_outputs,
        dtype=dtype,
        sequence_length=self.iterator.source_sequence_length,
        time_major=self.time_major)

    # Use the top layer for now
    self.encoder_state_list = [encoder_outputs]

    return encoder_state, encoder_outputs

  def _build_individual_encoder_layers(self, bi_encoder_outputs,
                                       num_uni_layers, dtype, hparams):
    """Run each of the encoder layer separately, not used in general seq2seq."""
    uni_cell_lists = model_helper._cell_list(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=num_uni_layers,
        num_residual_layers=self.num_encoder_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=self.num_gpus,
        base_gpu=1,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    encoder_inp = bi_encoder_outputs
    encoder_states = []
    self.encoder_state_list = [bi_encoder_outputs[:, :, :hparams.num_units],
                               bi_encoder_outputs[:, :, hparams.num_units:]]
    with tf.variable_scope("rnn/multi_rnn_cell"):
      for i, cell in enumerate(uni_cell_lists):
        with tf.variable_scope("cell_%d" % i) as scope:
          encoder_inp, encoder_state = tf.nn.dynamic_rnn(
              cell,
              encoder_inp,
              dtype=dtype,
              sequence_length=self.iterator.source_sequence_length,
              time_major=self.time_major,
              scope=scope)
          encoder_states.append(encoder_state)
          self.encoder_state_list.append(encoder_inp)

    encoder_state = tuple(encoder_states)
    encoder_outputs = self.encoder_state_list[-1]
    return encoder_state, encoder_outputs

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build a RNN cell with GNMT attention architecture."""
    # Standard attention
    if not self.is_gnmt_attention:
      return super(GNMTModel, self)._build_decoder_cell(
          hparams, encoder_outputs, encoder_state, source_sequence_length)

    # GNMT attention
    attention_option = hparams.attention
    attention_architecture = hparams.attention_architecture
    num_units = hparams.num_units
    infer_mode = hparams.infer_mode

    dtype = tf.float32

    if self.time_major:
      memory = tf.transpose(encoder_outputs, [1, 0, 2])
    else:
      memory = encoder_outputs

    if (self.mode == tf.contrib.learn.ModeKeys.INFER and
        infer_mode == "beam_search"):
      memory, source_sequence_length, encoder_state, batch_size = (
          self._prepare_beam_search_decoder_inputs(
              hparams.beam_width, memory, source_sequence_length,
              encoder_state))
    else:
      batch_size = self.batch_size

    devices = cluster_utils.get_pipeline_devices(hparams.pipeline_device_num)
    with tf.device(devices[0]):
      attention_mechanism = self.attention_mechanism_fn(
          attention_option, num_units, memory, source_sequence_length, self.mode)

    ## Dapple guard
    DAPPLE_TEST = hparams.dapple_test
    if not DAPPLE_TEST:
      cell_list = model_helper._cell_list(  # pylint: disable=protected-access
          unit_type=hparams.unit_type,
          num_units=num_units,
          num_layers=self.num_decoder_layers,
          num_residual_layers=self.num_decoder_residual_layers,
          forget_bias=hparams.forget_bias,
          dropout=hparams.dropout,
          num_gpus=self.num_gpus,
          base_gpu=1,
          mode=self.mode,
          single_cell_fn=self.single_cell_fn,
          residual_fn=gnmt_residual_fn
      )
    else:
      encoder_device=devices[0]
      decoder_device=devices[1]
      # Decoder layer0 (Add residual connection from layer3)
      uc0 = model_helper._single_cell_dapple(hparams, self.mode, 0, False, encoder_device, gnmt_residual_fn)
      uc1 = model_helper._single_cell_dapple(hparams, self.mode, 1, False, decoder_device, gnmt_residual_fn)
      uc2 = model_helper._single_cell_dapple(hparams, self.mode, 2, True, decoder_device, gnmt_residual_fn)
      uc3 = model_helper._single_cell_dapple(hparams, self.mode, 3, True, decoder_device, gnmt_residual_fn)
      uc4 = model_helper._single_cell_dapple(hparams, self.mode, 4, True, decoder_device, gnmt_residual_fn)
      uc5 = model_helper._single_cell_dapple(hparams, self.mode, 5, True, decoder_device, gnmt_residual_fn)
      uc6 = model_helper._single_cell_dapple(hparams, self.mode, 6, True, decoder_device, gnmt_residual_fn)
      uc7 = model_helper._single_cell_dapple(hparams, self.mode, 7, True, decoder_device, gnmt_residual_fn)
      cell_list = [uc0, uc1, uc2, uc3, uc4, uc5, uc6, uc7]
      if hparams.gnmt16: ## 16 encoder + 16 decoder
        uc8 = model_helper._single_cell_dapple(hparams, self.mode, 8, True,   decoder_device, gnmt_residual_fn)
        uc9 = model_helper._single_cell_dapple(hparams, self.mode, 9, True,   decoder_device, gnmt_residual_fn)
        uc10 = model_helper._single_cell_dapple(hparams, self.mode, 10, True, decoder_device, gnmt_residual_fn)
        uc11 = model_helper._single_cell_dapple(hparams, self.mode, 11, True, decoder_device, gnmt_residual_fn)
        uc12 = model_helper._single_cell_dapple(hparams, self.mode, 12, True, decoder_device, gnmt_residual_fn)
        uc13 = model_helper._single_cell_dapple(hparams, self.mode, 13, True, decoder_device, gnmt_residual_fn)
        uc14 = model_helper._single_cell_dapple(hparams, self.mode, 14, True, decoder_device, gnmt_residual_fn)
        uc15 = model_helper._single_cell_dapple(hparams, self.mode, 15, True, decoder_device, gnmt_residual_fn)
        cell_list = cell_list + [uc8, uc9, uc10, uc11, uc12, uc13, uc14, uc15]
      attention_cell = cell_list.pop(0)
      attention_cell = tf.contrib.seq2seq.AttentionWrapper(
          attention_cell,
          attention_mechanism,
          attention_layer_size=None,  # don't use attention layer.
          output_attention=False,
          alignment_history=False,
          name="attention")
      assert attention_architecture == "gnmt_v2"
      DAPPLE_TEST=True
      if DAPPLE_TEST:
        cell1 = GNMTAttentionMultiCell_1(
          attention_cell, [], hparams.batch_size, use_new_attention=True)
        cell2 = GNMTAttentionMultiCell_2(
          cell_list, use_new_attention=True,
          global_attention=cell1.global_attention)

        with tf.device(devices[0]):
          zs1 = cell1.zero_state(batch_size, dtype)
        with tf.device(devices[1]):
         zs2 = cell2.zero_state(batch_size, dtype)
        decoder_initial_state = [zs1, zs2]
        cell = [cell1, cell2]
      else:
        cell = GNMTAttentionMultiCell(
          attention_cell, cell_list, use_new_attention=True)
        decoder_initial_state = cell.zero_state(batch_size, dtype)
      return cell, decoder_initial_state

    # Only wrap the bottom layer with the attention mechanism.
    attention_cell = cell_list.pop(0)

    # Only generate alignment in greedy INFER mode.
    alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                         infer_mode != "beam_search")
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(
        attention_cell,
        attention_mechanism,
        attention_layer_size=None,  # don't use attention layer.
        output_attention=False,
        alignment_history=alignment_history,
        name="attention")

    if attention_architecture == "gnmt":
      cell = GNMTAttentionMultiCell(
          attention_cell, cell_list)
    elif attention_architecture == "gnmt_v2":
      cell = GNMTAttentionMultiCell(
          attention_cell, cell_list, use_new_attention=True)
    else:
      raise ValueError(
          "Unknown attention_architecture %s" % attention_architecture)

    if hparams.pass_hidden_state:
      decoder_initial_state = tuple(
          zs.clone(cell_state=es)
          if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
          for zs, es in zip(
              cell.zero_state(batch_size, dtype), encoder_state))
    else:
      decoder_initial_state = cell.zero_state(batch_size, dtype)

    return cell, decoder_initial_state

  def _get_infer_summary(self, hparams):
    if hparams.infer_mode == "beam_search":
      return tf.no_op()
    elif self.is_gnmt_attention:
      return attention_model._create_attention_images_summary(
          self.final_context_state[0])
    else:
      return super(GNMTModel, self)._get_infer_summary(hparams)

#MAX_INDEX=338
MAX_INDEX=1
class GNMTAttentionMultiCell_1(tf.nn.rnn_cell.MultiRNNCell):
  """A MultiCell with GNMT attention style."""

  def __init__(self, attention_cell, cells, batch_size=32, use_new_attention=False):
    """Creates a GNMTAttentionMultiCell.

    Args:
      attention_cell: An instance of AttentionWrapper.
      cells: A list of RNNCell wrapped with AttentionInputWrapper.
      use_new_attention: Whether to use the attention generated from current
        step bottom layer's output. Default is False.
    """
    cells = [attention_cell] + cells
    self.use_new_attention = use_new_attention
    self.global_attention = []
    self.loop1_index = 0
    devices = cluster_utils.get_pipeline_devices(2)
    with tf.variable_scope("global_attention"), tf.device(devices[0]):
      for i in range(MAX_INDEX):
        self.global_attention.append(tf.Variable(np.zeros((batch_size, 1024)), dtype=tf.float32,
            trainable=False, name="global_attention_%d" % i))
    super(GNMTAttentionMultiCell_1, self).__init__(cells, state_is_tuple=True)

  def __call__(self, inputs, state, scope=None):
    """Run the cell with bottom layer's attention copied to all upper layers."""
    if not tf.contrib.framework.nest.is_sequence(state):
      raise ValueError(
          "Expected state to be a tuple of length %d, but received: %s"
          % (len(self.state_size), state))

    devices = cluster_utils.get_pipeline_devices(2)
    with tf.variable_scope(scope or "multi_rnn_cell_part1", reuse=tf.AUTO_REUSE), tf.device(devices[0]):
      new_states = []

      with tf.variable_scope("cell_0_attention"):
        attention_cell = self._cells[0]
        attention_state = state[0]
        cur_inp, new_attention_state = attention_cell(inputs, attention_state)
        new_states.append(new_attention_state)

      assign_op = tf.assign(self.global_attention[self.loop1_index], new_attention_state.attention,
          validate_shape=False, name="global_attention_assign")
      with tf.control_dependencies([assign_op]):
        self.loop1_index += 1
      if self.loop1_index >= MAX_INDEX:
        self.loop1_index = 0

    return cur_inp, tuple(new_states)


class GNMTAttentionMultiCell_2(tf.nn.rnn_cell.MultiRNNCell):
  """A MultiCell with GNMT attention style."""

  def __init__(self, cells, use_new_attention=False, global_attention=None):
    """Creates a GNMTAttentionMultiCell.

    Args:
      attention_cell: An instance of AttentionWrapper.
      cells: A list of RNNCell wrapped with AttentionInputWrapper.
      use_new_attention: Whether to use the attention generated from current
        step bottom layer's output. Default is False.
    """
    self.use_new_attention = use_new_attention
    self.global_attention = global_attention
    self.loop2_index = 0
    super(GNMTAttentionMultiCell_2, self).__init__(cells, state_is_tuple=True)

  def __call__(self, inputs, state, scope=None):
    """Run the cell with bottom layer's attention copied to all upper layers."""
    if not tf.contrib.framework.nest.is_sequence(state):
      raise ValueError(
          "Expected state to be a tuple of length %d, but received: %s"
          % (len(self.state_size), state))

    cur_inp = inputs
    devices = cluster_utils.get_pipeline_devices(2)
    with tf.variable_scope(scope or "multi_rnn_cell_part2", reuse=tf.AUTO_REUSE), tf.device(devices[1]):
      new_attention_state = state[0]
      new_states = []


      new_attention = self.global_attention[self.loop2_index]
      for i in range(0, len(self._cells)):
        #with tf.variable_scope("cell_%d" % (i+2)):
        with tf.variable_scope("cell_%d" % (i+1)):
          cell = self._cells[i]
          cur_state = state[i]

          cur_inp = tf.concat([cur_inp, new_attention], -1)
          ## QQQQQQ Temp work around 11-29
          #cur_inp = tf.concat([cur_inp, cur_inp], -1)

          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
      self.loop2_index += 1
      if self.loop2_index >= MAX_INDEX:
        self.loop2_index = 0

      return cur_inp, tuple(new_states)

class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
  """A MultiCell with GNMT attention style."""

  def __init__(self, attention_cell, cells, use_new_attention=False):
    """Creates a GNMTAttentionMultiCell.

    Args:
      attention_cell: An instance of AttentionWrapper.
      cells: A list of RNNCell wrapped with AttentionInputWrapper.
      use_new_attention: Whether to use the attention generated from current
        step bottom layer's output. Default is False.
    """
    cells = [attention_cell] + cells
    self.use_new_attention = use_new_attention
    super(GNMTAttentionMultiCell, self).__init__(cells, state_is_tuple=True)

  def __call__(self, inputs, state, scope=None):
    """Run the cell with bottom layer's attention copied to all upper layers."""
    if not tf.contrib.framework.nest.is_sequence(state):
      raise ValueError(
          "Expected state to be a tuple of length %d, but received: %s"
          % (len(self.state_size), state))

    with tf.variable_scope(scope or "multi_rnn_cell", reuse=tf.AUTO_REUSE):
      new_states = []

      devices = cluster_utils.get_pipeline_devices(2)
      with tf.variable_scope("cell_0_attention"), tf.device(devices[0]):
        attention_cell = self._cells[0]
        attention_state = state[0]
        cur_inp, new_attention_state = attention_cell(inputs, attention_state)
        new_states.append(new_attention_state)

      for i in range(1, len(self._cells)):
        with tf.variable_scope("cell_%d" % i):

          cell = self._cells[i]
          cur_state = state[i]

          if self.use_new_attention:
            cur_inp = tf.concat([cur_inp, new_attention_state.attention], -1)
          else:
            cur_inp = tf.concat([cur_inp, attention_state.attention], -1)

          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)

    return cur_inp, tuple(new_states)


def gnmt_residual_fn(inputs, outputs):
  """Residual function that handles different inputs and outputs inner dims.

  Args:
    inputs: cell inputs, this is actual inputs concatenated with the attention
      vector.
    outputs: cell outputs

  Returns:
    outputs + actual inputs
  """
  def split_input(inp, out):
    out_dim = out.get_shape().as_list()[-1]
    inp_dim = inp.get_shape().as_list()[-1]
    return tf.split(inp, [out_dim, inp_dim - out_dim], axis=-1)
  actual_inputs, _ = tf.contrib.framework.nest.map_structure(
      split_input, inputs, outputs)
  def assert_shape_match(inp, out):
    inp.get_shape().assert_is_compatible_with(out.get_shape())
  tf.contrib.framework.nest.assert_same_structure(actual_inputs, outputs)
  tf.contrib.framework.nest.map_structure(
      assert_shape_match, actual_inputs, outputs)
  return tf.contrib.framework.nest.map_structure(
      lambda inp, out: inp + out, actual_inputs, outputs)
