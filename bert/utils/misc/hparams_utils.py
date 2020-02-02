from __future__ import print_function

import os
import tensorflow as tf

import vocab_utils, misc_utils as utils


def create_hparams(flags):
  """Create training hparams."""
  default_hparams = tf.contrib.training.HParams(
      # Data
      src=flags.src,
      tgt=flags.tgt,
      train_prefix=flags.train_prefix,
      dev_prefix=flags.dev_prefix,
      test_prefix=flags.test_prefix,
      vocab_prefix=flags.vocab_prefix,
      embed_prefix=flags.embed_prefix,

      # Networks
      num_units=flags.num_units,
      num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
      num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
      dropout=flags.dropout,
      unit_type=flags.unit_type,
      encoder_type=flags.encoder_type,
      residual=flags.residual,
      time_major=flags.time_major,
      num_embeddings_partitions=flags.num_embeddings_partitions,

      # Attention mechanisms
      attention=flags.attention,
      attention_architecture=flags.attention_architecture,
      output_attention=flags.output_attention,
      pass_hidden_state=flags.pass_hidden_state,

      # Train
      optimizer=flags.optimizer,
      stop_at_step=flags.stop_at_step,
      batch_size=flags.batch_size,
      init_op=flags.init_op,
      init_weight=flags.init_weight,
      max_gradient_norm=flags.max_gradient_norm,
      learning_rate=flags.learning_rate,
      warmup_steps=flags.warmup_steps,
      warmup_scheme=flags.warmup_scheme,
      decay_scheme=flags.decay_scheme,
      colocate_gradients_with_ops=flags.colocate_gradients_with_ops,
      num_sampled_softmax=flags.num_sampled_softmax,

      # Data constraints
      num_buckets=flags.num_buckets,
      max_train=flags.max_train,
      src_max_len=flags.src_max_len,
      tgt_max_len=flags.tgt_max_len,

      # Vocab
      sos=flags.sos if flags.sos else vocab_utils.SOS,
      eos=flags.eos if flags.eos else vocab_utils.EOS,
      subword_option=flags.subword_option,
      check_special_token=flags.check_special_token,
      use_char_encode=flags.use_char_encode,

      # Misc
      forget_bias=flags.forget_bias,
      epoch_step=0,  # record where we were within an epoch.
      log_loss_every_n_iters=flags.log_loss_every_n_iters,
      steps_per_external_eval=flags.steps_per_external_eval,
      share_vocab=flags.share_vocab,
      metrics=flags.metrics.split(","),
      log_device_placement=flags.log_device_placement,
      random_seed=flags.random_seed,
      override_loaded_hparams=flags.override_loaded_hparams,
      num_keep_ckpts=flags.num_keep_ckpts,
      avg_ckpts=flags.avg_ckpts,
      language_model=flags.language_model,
      intra_op_parallelism_threads=flags.intra_op_parallelism_threads,
      inter_op_parallelism_threads=flags.inter_op_parallelism_threads,
  )

  # Load hparams.
  loaded_hparams = False
  if not loaded_hparams:  # Try to load from out_dir
    hparams = create_or_load_hparams(default_hparams, flags.hparams_path, save_hparams=False)
  return hparams


def _add_argument(hparams, key, value, update=True):
  """Add an argument to hparams; if exists, change the value if update==True."""
  if hasattr(hparams, key):
    if update:
      setattr(hparams, key, value)
  else:
    hparams.add_hparam(key, value)


def extend_hparams(hparams):
  """Add new arguments to hparams."""
  # Sanity checks
  if hparams.encoder_type == "bi" and hparams.num_encoder_layers % 2 != 0:
    raise ValueError("For bi, num_encoder_layers %d should be even" %
                     hparams.num_encoder_layers)
  if (hparams.attention_architecture in ["gnmt"] and
      hparams.num_encoder_layers < 2):
    raise ValueError("For gnmt attention architecture, "
                     "num_encoder_layers %d should be >= 2" %
                     hparams.num_encoder_layers)
  if hparams.subword_option and hparams.subword_option not in ["spm", "bpe"]:
    raise ValueError("subword option must be either spm, or bpe")

  # Different number of encoder / decoder layers
  assert hparams.num_encoder_layers and hparams.num_decoder_layers
  if hparams.num_encoder_layers != hparams.num_decoder_layers:
    hparams.pass_hidden_state = False
    utils.print_out("Num encoder layer %d is different from num decoder layer"
                    " %d, so set pass_hidden_state to False" % (
                        hparams.num_encoder_layers,
                        hparams.num_decoder_layers))

  # Set residual layers
  num_encoder_residual_layers = 0
  num_decoder_residual_layers = 0
  if hparams.residual:
    if hparams.num_encoder_layers > 1:
      num_encoder_residual_layers = hparams.num_encoder_layers - 1
    if hparams.num_decoder_layers > 1:
      num_decoder_residual_layers = hparams.num_decoder_layers - 1

    if hparams.encoder_type == "gnmt":
      # The first unidirectional layer (after the bi-directional layer) in
      # the GNMT encoder can't have residual connection due to the input is
      # the concatenation of fw_cell and bw_cell's outputs.
      num_encoder_residual_layers = hparams.num_encoder_layers - 2

      # Compatible for GNMT models
      if hparams.num_encoder_layers == hparams.num_decoder_layers:
        num_decoder_residual_layers = num_encoder_residual_layers
  _add_argument(hparams, "num_encoder_residual_layers",
                num_encoder_residual_layers)
  _add_argument(hparams, "num_decoder_residual_layers",
                num_decoder_residual_layers)

  # Language modeling
  if getattr(hparams, "language_model", None):
    hparams.attention = ""
    hparams.attention_architecture = ""
    hparams.pass_hidden_state = False
    hparams.share_vocab = True
    hparams.src = hparams.tgt
    utils.print_out("For language modeling, we turn off attention and "
                    "pass_hidden_state; turn on share_vocab; set src to tgt.")

  # Source vocab
  check_special_token = getattr(hparams, "check_special_token", True)
  src_vocab_size, src_vocab_file = 7710, None
  #vocab_utils.check_vocab(
  #    src_vocab_file,
  #    check_special_token=check_special_token,
  #    sos=hparams.sos,
  #    eos=hparams.eos,
  #    unk=vocab_utils.UNK)

  # Target vocab
  if hparams.share_vocab:
    utils.print_out("  using source vocab for target")
    tgt_vocab_file = src_vocab_file
    tgt_vocab_size = src_vocab_size
  else:
    tgt_vocab_size, tgt_vocab_file = 17192, None
    #  vocab_utils.check_vocab(
    #    tgt_vocab_file,
    #    check_special_token=check_special_token,
    #    sos=hparams.sos,
    #    eos=hparams.eos,
    #    unk=vocab_utils.UNK)
  _add_argument(hparams, "src_vocab_size", src_vocab_size)
  _add_argument(hparams, "tgt_vocab_size", tgt_vocab_size)
  _add_argument(hparams, "src_vocab_file", src_vocab_file)
  _add_argument(hparams, "tgt_vocab_file", tgt_vocab_file)

  # Num embedding partitions
  num_embeddings_partitions = getattr(hparams, "num_embeddings_partitions", 0)
  _add_argument(hparams, "num_enc_emb_partitions", num_embeddings_partitions)
  _add_argument(hparams, "num_dec_emb_partitions", num_embeddings_partitions)

  # Pretrained Embeddings
  _add_argument(hparams, "src_embed_file", "")
  _add_argument(hparams, "tgt_embed_file", "")
  if getattr(hparams, "embed_prefix", None):
    src_embed_file = hparams.embed_prefix + "." + hparams.src
    tgt_embed_file = hparams.embed_prefix + "." + hparams.tgt

    if tf.gfile.Exists(src_embed_file):
      utils.print_out("  src_embed_file %s exist" % src_embed_file)
      hparams.src_embed_file = src_embed_file

      utils.print_out(
          "For pretrained embeddings, set num_enc_emb_partitions to 1")
      hparams.num_enc_emb_partitions = 1
    else:
      utils.print_out("  src_embed_file %s doesn't exist" % src_embed_file)

    if tf.gfile.Exists(tgt_embed_file):
      utils.print_out("  tgt_embed_file %s exist" % tgt_embed_file)
      hparams.tgt_embed_file = tgt_embed_file

      utils.print_out(
          "For pretrained embeddings, set num_dec_emb_partitions to 1")
      hparams.num_dec_emb_partitions = 1
    else:
      utils.print_out("  tgt_embed_file %s doesn't exist" % tgt_embed_file)

  return hparams


def ensure_compatible_hparams(hparams, default_hparams, hparams_path=""):
  """Make sure the loaded hparams is compatible with new changes."""
  default_hparams = utils.maybe_parse_standard_hparams(
      default_hparams, hparams_path)

  # Set num encoder/decoder layers (for old checkpoints)
  if hasattr(hparams, "num_layers"):
    if not hasattr(hparams, "num_encoder_layers"):
      hparams.add_hparam("num_encoder_layers", hparams.num_layers)
    if not hasattr(hparams, "num_decoder_layers"):
      hparams.add_hparam("num_decoder_layers", hparams.num_layers)

  # For compatible reason, if there are new fields in default_hparams,
  #   we add them to the current hparams
  default_config = default_hparams.values()
  config = hparams.values()
  for key in default_config:
    if key not in config:
      hparams.add_hparam(key, default_config[key])

  # Update all hparams' keys if override_loaded_hparams=True
  overwritten_keys = None
  if getattr(default_hparams, "override_loaded_hparams", None):
    overwritten_keys = default_config.keys()

  if overwritten_keys is not None:
    for key in overwritten_keys:
      if getattr(hparams, key) != default_config[key]:
        utils.print_out("# Updating hparams.%s: %s -> %s" %
                        (key, str(getattr(hparams, key)),
                         str(default_config[key])))
        setattr(hparams, key, default_config[key])
  return hparams


def create_or_load_hparams(default_hparams, hparams_path, save_hparams=True):
  """Create hparams or load hparams from out_dir."""
  hparams = None
  if not hparams:
    hparams = default_hparams
    hparams = utils.maybe_parse_standard_hparams(
        hparams, hparams_path)
  else:
    hparams = ensure_compatible_hparams(hparams, default_hparams, hparams_path)
  hparams = extend_hparams(hparams)

  # Print HParams
  utils.print_hparams(hparams)
  return hparams
