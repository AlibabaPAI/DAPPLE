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

"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf

from . import model_helper
from utils.misc import misc_utils as utils

utils.check_tensorflow_version()

__all__ = ["BaseModel", "Model"]


class BaseModel(object):
  """Sequence-to-sequence base class.
  """

  def __init__(self,
               hparams,
               mode,
               src_ids,
               src_seq_len,
               tgt_ids,
               tgt_seq_len,
               tgt_ids_out,
               batch_size=None,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      source_vocab_table: Lookup table mapping source words to ids.
      target_vocab_table: Lookup table mapping target words to ids.
      reverse_target_vocab_table: Lookup table mapping ids to target words. Only
        required in INFER mode. Defaults to None.
      scope: scope of the model.
      extra_args: model_helper.ExtraArgs, for passing customizable functions.

    """
    # Set params
    self._set_params_initializer(hparams, mode, src_ids, src_seq_len, tgt_ids, tgt_seq_len, tgt_ids_out,
                                 batch_size, scope, extra_args)

    # Not used in general seq2seq models; when True, ignore decoder & training
    self.extract_encoder_layers = (hasattr(hparams, "extract_encoder_layers")
                                   and hparams.extract_encoder_layers)

    # Train graph
    res = self.build_graph(hparams, scope=scope)
    if not self.extract_encoder_layers:
      self._set_train(res, reverse_target_vocab_table, hparams)

  def pad_in_time(self, x, padding_length):
    """Helper function to pad a tensor in the time dimension."""
    paddings = [[0, padding_length], [0, 0]]
    x = tf.pad(x, paddings)
    return x

  def _pad_in_time(self, x, padding_length):
    """Helper function to pad a tensor in the time dimension and retain the static depth dimension."""
    depth = x.get_shape().as_list()[-1]
    padding = [[0, padding_length], [0, 0], [0, 0]] if self.time_major else [[0, 0], [0, padding_length], [0, 0]]
    x = tf.pad(x, padding)
    x.set_shape((None, None, depth))
    return x

  def align_in_time(self, x, y):
    """Aligns the time dimension of :obj:`y` with :obj:`time_dim`."""
    time_dim = tf.shape(x)[0] if self.time_major else tf.shape(x)[1]
    length = tf.shape(y)[0]
    return tf.cond(
      tf.less(time_dim, length),
      true_fn=lambda: y[:time_dim, :],
      false_fn=lambda: self.pad_in_time(y, time_dim - length))

  def _align_in_time(self, x, y):
    """Aligns the time dimension of :obj:`x` with :obj:`length`."""
    time_dim = tf.shape(x)[0] if self.time_major else tf.shape(x)[1]
    length = tf.shape(y)[0]

    return tf.cond(
      tf.less(time_dim, length),
      true_fn=lambda: self._pad_in_time(x, length - time_dim),
      false_fn=lambda: x[:, :length])

  def _set_params_initializer(self,
                              hparams,
                              mode,
                              src_ids,
                              src_seq_len,
                              tgt_ids,
                              tgt_seq_len,
                              tgt_ids_out,
                              batch_size,
                              scope,
                              extra_args=None):
    """Set various params for self and initialize."""
    self.src_ids = src_ids
    self.src_seq_len = src_seq_len
    self.tgt_ids = tgt_ids
    self.tgt_seq_len = tgt_seq_len
    self.tgt_ids_out = tgt_ids_out
    self.mode = mode

    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
    self.time_major = hparams.time_major

    if hparams.use_char_encode:
      assert (not self.time_major), ("Can't use time major for"
                                     " char-level inputs.")

    self.dtype = tf.float32
    self.num_sampled_softmax = hparams.num_sampled_softmax

    # extra_args: to make it flexible for adding external customizable code
    self.single_cell_fn = None
    if extra_args:
      self.single_cell_fn = extra_args.single_cell_fn

    # Set num units
    self.num_units = hparams.num_units

    # Set num layers
    self.num_encoder_layers = hparams.num_encoder_layers
    self.num_decoder_layers = hparams.num_decoder_layers

    # Set num residual layers
    if hasattr(hparams, "num_residual_layers"):  # compatible common_test_utils
      self.num_encoder_residual_layers = hparams.num_residual_layers
      self.num_decoder_residual_layers = hparams.num_residual_layers
    else:
      self.num_encoder_residual_layers = hparams.num_encoder_residual_layers
      self.num_decoder_residual_layers = hparams.num_decoder_residual_layers

    # Batch size
    self.batch_size = batch_size

    # Initializer
    self.random_seed = hparams.random_seed
    initializer = model_helper.get_initializer(
      hparams.init_op, self.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    if extra_args and extra_args.encoder_emb_lookup_fn:
      self.encoder_emb_lookup_fn = extra_args.encoder_emb_lookup_fn
    else:
      self.encoder_emb_lookup_fn = tf.nn.embedding_lookup
    self.init_embeddings(hparams, scope)

  def _set_train(self, res, reverse_target_vocab_table, hparams):
    """Set up training"""
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[1]

  def init_embeddings(self, hparams, scope):
    """Init embeddings."""
    self.embedding_encoder, self.embedding_decoder = (
      model_helper.create_emb_for_encoder_and_decoder(
        share_vocab=hparams.share_vocab,
        src_vocab_size=self.src_vocab_size,
        tgt_vocab_size=self.tgt_vocab_size,
        src_embed_size=self.num_units,
        tgt_embed_size=self.num_units,
        num_enc_partitions=hparams.num_enc_emb_partitions,
        num_dec_partitions=hparams.num_dec_emb_partitions,
        src_vocab_file=hparams.src_vocab_file,
        tgt_vocab_file=hparams.tgt_vocab_file,
        src_embed_file=hparams.src_embed_file,
        tgt_embed_file=hparams.tgt_embed_file,
        use_char_encode=hparams.use_char_encode,
        scope=scope, ))

  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, loss_tuple, final_context_state, sample_id),
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols].
        loss: loss = the total loss / batch_size.
        final_context_state: the final state of decoder RNN.
        sample_id: sampling indices.

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    utils.print_out("# Creating %s graph ..." % self.mode)

    # Projection
    if not self.extract_encoder_layers:
      with tf.variable_scope(scope or "build_network"):
        with tf.variable_scope("decoder/output_projection"):
          self.output_layer = tf.layers.Dense(
            self.tgt_vocab_size, use_bias=False, name="output_projection")

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=self.dtype):
      # Encoder
      if hparams.language_model:  # no encoder for language modeling
        utils.print_out("  language modeling: no encoder")
        self.encoder_outputs = None
        encoder_state = None
      else:
        self.encoder_outputs, encoder_state = self._build_encoder(hparams)

      # Skip decoder if extracting only encoder layers
      if self.extract_encoder_layers:
        return

      ## Decoder
      logits, decoder_cell_outputs, sample_id, final_context_state = (
        self._build_decoder(self.encoder_outputs, encoder_state, hparams))

      ## Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        loss = self._compute_loss(logits, decoder_cell_outputs)
      else:
        loss = tf.constant(0.0)

      return logits, loss, final_context_state, sample_id

  @abc.abstractmethod
  def _build_encoder(self, hparams):
    """Subclass must implement this.

    Build and run an RNN encoder.

    Args:
      hparams: Hyperparameters configurations.

    Returns:
      A tuple of encoder_outputs and encoder_state.
    """
    pass

  def _build_encoder_cell(self, hparams, num_layers, num_residual_layers):
    """Build a multi-layer RNN cell that can be used by encoder."""

    return model_helper.create_rnn_cell(
      unit_type=hparams.unit_type,
      num_units=self.num_units,
      num_layers=num_layers,
      num_residual_layers=num_residual_layers,
      forget_bias=hparams.forget_bias,
      dropout=hparams.dropout,
      mode=self.mode,
      single_cell_fn=self.single_cell_fn)

  def _build_decoder(self, encoder_outputs, encoder_state, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """

    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope:
      cell, decoder_initial_state = self._build_decoder_cell(
        hparams, encoder_outputs, encoder_state,
        self.src_seq_len)

      # Optional ops depends on which mode we are in and which loss function we
      # are using.
      logits = tf.no_op()
      decoder_cell_outputs = None

      ## Train
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        # decoder_emp_inp: [max_time, batch_size, num_units]
        target_input = self.tgt_ids
        if self.time_major:
          target_input = tf.transpose(target_input)
        decoder_emb_inp = tf.nn.embedding_lookup(
          self.embedding_decoder, target_input)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
          decoder_emb_inp, self.tgt_seq_len,
          time_major=self.time_major)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
          cell,
          helper,
          decoder_initial_state, )

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
          my_decoder,
          output_time_major=self.time_major,
          swap_memory=True,
          scope=decoder_scope)

        sample_id = outputs.sample_id

        if self.num_sampled_softmax > 0:
          # Note: this is required when using sampled_softmax_loss.
          decoder_cell_outputs = outputs.rnn_output

        # Colocate output layer with the last RNN cell if there is no extra GPU
        # available. Otherwise, put last layer on a separate GPU.
        logits = self.output_layer(outputs.rnn_output)

        if self.num_sampled_softmax > 0:
          logits = tf.no_op()  # unused when using sampled softmax loss.

    return logits, decoder_cell_outputs, sample_id, final_context_state

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  @abc.abstractmethod
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Subclass must implement this.

    Args:
      hparams: Hyperparameters configurations.
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      source_sequence_length: sequence length of encoder_outputs.

    Returns:
      A tuple of a multi-layer RNN cell used by decoder and the intial state of
      the decoder RNN.
    """
    pass

  def _softmax_cross_entropy_loss(
      self, logits, decoder_cell_outputs, labels):
    """Compute softmax loss or sampled softmax loss."""
    if self.num_sampled_softmax > 0:

      is_sequence = (decoder_cell_outputs.shape.ndims == 3)

      if is_sequence:
        labels = tf.reshape(labels, [-1, 1])
        inputs = tf.reshape(decoder_cell_outputs, [-1, self.num_units])

      crossent = tf.nn.sampled_softmax_loss(
        weights=tf.transpose(self.output_layer.kernel),
        biases=self.output_layer.bias or tf.zeros([self.tgt_vocab_size]),
        labels=labels,
        inputs=inputs,
        num_sampled=self.num_sampled_softmax,
        num_classes=self.tgt_vocab_size,
        partition_strategy="div",
        seed=self.random_seed)

      if is_sequence:
        if self.time_major:
          crossent = tf.reshape(crossent, [-1, self.batch_size])
        else:
          crossent = tf.reshape(crossent, [self.batch_size, -1])

    else:
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)

    return crossent

  def _compute_loss(self, logits, decoder_cell_outputs):
    """Compute optimization loss."""
    target_output = self.tgt_ids_out
    if self.time_major:
      target_output = tf.transpose(target_output)

    # Make sure outputs have the same time_dim as labels
    target_output = self.align_in_time(logits, target_output)
    max_time = self.get_max_time(target_output)

    crossent = self._softmax_cross_entropy_loss(
      logits, decoder_cell_outputs, target_output)

    target_weights = tf.sequence_mask(
      self.tgt_seq_len, max_time, dtype=self.dtype)
    if self.time_major:
      target_weights = tf.transpose(target_weights)

    loss = tf.reduce_sum(
      crossent * target_weights) / tf.to_float(self.batch_size)
    return loss


class Model(BaseModel):
  """Sequence-to-sequence dynamic model.

  This class implements a multi-layer recurrent neural network as encoder,
  and a multi-layer recurrent neural network decoder.
  """

  def _build_encoder_from_sequence(self, hparams, sequence, sequence_length):
    """Build an encoder from a sequence.

    Args:
      hparams: hyperparameters.
      sequence: tensor with input sequence data.
      sequence_length: tensor with length of the input sequence.

    Returns:
      encoder_outputs: RNN encoder outputs.
      encoder_state: RNN encoder state.

    Raises:
      ValueError: if encoder_type is neither "uni" nor "bi".
    """
    num_layers = self.num_encoder_layers
    num_residual_layers = self.num_encoder_residual_layers

    if self.time_major:
      sequence = tf.transpose(sequence)

    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype

      self.encoder_emb_inp = self.encoder_emb_lookup_fn(
        self.embedding_encoder, sequence)

      # Encoder_outputs: [max_time, batch_size, num_units]
      if hparams.encoder_type == "uni":
        utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                        (num_layers, num_residual_layers))
        cell = self._build_encoder_cell(hparams, num_layers,
                                        num_residual_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
          cell,
          self.encoder_emb_inp,
          dtype=dtype,
          sequence_length=sequence_length,
          time_major=self.time_major,
          swap_memory=True)
      elif hparams.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)
        utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                        (num_bi_layers, num_bi_residual_layers))

        encoder_outputs, bi_encoder_state = (
          self._build_bidirectional_rnn(
            inputs=self.encoder_emb_inp,
            sequence_length=sequence_length,
            dtype=dtype,
            hparams=hparams,
            num_bi_layers=num_bi_layers,
            num_bi_residual_layers=num_bi_residual_layers))

        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

    # Use the top layer for now
    self.encoder_state_list = [encoder_outputs]

    return encoder_outputs, encoder_state

  def _build_encoder(self, hparams):
    """Build encoder from source."""
    utils.print_out("# Build a basic encoder")
    with tf.variable_scope('BasicEncoderBuilt'):
      return self._build_encoder_from_sequence(
        hparams, self.src_ids, self.src_seq_len)

  def _build_bidirectional_rnn(self, inputs, sequence_length,
                               dtype, hparams,
                               num_bi_layers,
                               num_bi_residual_layers):
    """Create and call biddirectional RNN cells.

    Args:
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
        layers in each RNN cell will be wrapped with `ResidualWrapper`.
    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    fw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers)
    bw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers)

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
      fw_cell,
      bw_cell,
      inputs,
      dtype=dtype,
      sequence_length=sequence_length,
      time_major=self.time_major,
      swap_memory=True)

    return tf.concat(bi_outputs, -1), bi_state

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build an RNN cell that can be used by decoder."""
    # We only make use of encoder_outputs in attention-based models
    if hparams.attention:
      raise ValueError("BasicModel doesn't support attention.")

    cell = model_helper.create_rnn_cell(
      unit_type=hparams.unit_type,
      num_units=self.num_units,
      num_layers=self.num_decoder_layers,
      num_residual_layers=self.num_decoder_residual_layers,
      forget_bias=hparams.forget_bias,
      dropout=hparams.dropout,
      mode=self.mode,
      single_cell_fn=self.single_cell_fn
    )

    if hparams.language_model:
      encoder_state = cell.zero_state(self.batch_size, self.dtype)
    elif not hparams.pass_hidden_state:
      raise ValueError("For non-attentional model, "
                       "pass_hidden_state needs to be set to True")

    # For beam search, we need to replicate encoder infos beam_width times
    if (self.mode == tf.contrib.learn.ModeKeys.INFER and
        hparams.infer_mode == "beam_search"):
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=hparams.beam_width)
    else:
      decoder_initial_state = encoder_state

    return cell, decoder_initial_state
