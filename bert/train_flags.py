import tensorflow as tf

# dataset
tf.app.flags.DEFINE_string('dataset_name', 'mock_iwslt15',
                           'default mock_iwslt15 for NMT.')
tf.app.flags.DEFINE_string('task_type', 'train', 'train or evaluation, default train.')
tf.app.flags.DEFINE_string('dataset_dir', None, '')
tf.app.flags.DEFINE_string('file_pattern', "tf_record", 'file pattern for dataset, eg. tfrecord')
tf.app.flags.DEFINE_integer('num_sample_per_epoch', 1000000, '')
tf.app.flags.DEFINE_string("train_volumes_files", None,
                           "This specifies volume files to train.")
tf.app.flags.DEFINE_string("eval_volumes_files", None,
                           "This specifies volume files to eval.")

# preprocessing
tf.app.flags.DEFINE_integer(
  'num_preprocessing_threads', 16,
  'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
  'shuffle_buffer_size', 1024, '')
tf.app.flags.DEFINE_integer(
  'prefetch_buffer_size', 32, '')
tf.app.flags.DEFINE_integer(
  'num_parallel_batches', 8, '')


# Model base parameters
tf.app.flags.DEFINE_string(
  'model_name', '', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string("model_type", "mrc",
                           "Model types could be one of [classification, regression, pretrain, mrc], "
                           "default is pretrain.")
tf.app.flags.DEFINE_string("model_dir", "",
                    "The path corresponding to the pre-trained BERT model.")
tf.app.flags.DEFINE_bool("do_lower_case", True,
                         "Whether to lower case the input text. Should be True for uncased "
                         "models and False for cased models.")
tf.app.flags.DEFINE_integer("max_seq_length", 384,
                            "The maximum total input sequence length after WordPiece tokenization. "
                            "Sequences longer than this will be truncated, and sequences shorter "
                            "than this will be padded.")
tf.app.flags.DEFINE_integer("mock_seq_length", 20,
                            "The maximum total input sequence length after WordPiece tokenization. "
                            "Sequences longer than this will be truncated, and sequences shorter "
                            "than this will be padded.")
tf.app.flags.DEFINE_bool(
  "do_train", True, "Whether to run training.")
tf.app.flags.DEFINE_bool(
  "do_predict", True, "Whether to run eval on the dev set.")
tf.app.flags.DEFINE_string(
  'loss_name', 'example', 'loss file name, see losses/ for more details.')
tf.app.flags.DEFINE_integer(
  'num_epochs', 100, '')
tf.app.flags.DEFINE_float(
  'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_integer(
  'batch_size', 32, 'batch size for training.')
tf.app.flags.DEFINE_integer("predict_batch_size", 32,
                            "batch size for predictions.")
tf.app.flags.DEFINE_bool('linear_warmup', False,
                         'if global_step < num_warmup_steps, the'
                         'learning rate will be `global_step/num_warmup_steps * init_lr`.')
tf.app.flags.DEFINE_float("warmup_proportion", 0.1,
                          "Proportion of training to perform linear learning rate warmup for. "
                          "E.g., 0.1 = 10% of training.")
tf.app.flags.DEFINE_string('job_name', None, 'job_name')

# Machine parameter
tf.app.flags.DEFINE_integer('task_index', 0, 'task_index')
tf.app.flags.DEFINE_string('taskId', None, '')
tf.app.flags.DEFINE_string('worker_hosts', None, 'worker_hosts')
tf.app.flags.DEFINE_string('ps_hosts', None, '')
tf.app.flags.DEFINE_integer('worker_count', None, '')
tf.app.flags.DEFINE_integer('ps_count', None, '')

# Learning rate tuning parameters
tf.app.flags.DEFINE_float(
  'num_epochs_per_decay', 2.0,
  'Number of epochs after which learning rate decays. Note: this flag counts '
  'epochs per clone but aggregates per sync replicas. So 1.0 means that '
  'each clone will go over full epoch individually, but replicas will go '
  'once across all replicas.')
tf.app.flags.DEFINE_bool(
  'sync_replicas', False,
  'Whether or not to synchronize the replicas during training.')
tf.app.flags.DEFINE_integer(
  'replicas_to_aggregate', 1,
  'The Number of gradients to collect before updating params.')
tf.app.flags.DEFINE_string(
  'learning_rate_decay_type',
  'exponential',
  'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
  ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
  'end_learning_rate', 0.0001,
  'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
  'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
  'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

# Optimizer parameters
tf.app.flags.DEFINE_string(
  'optimizer', 'rmsprop',
  'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
  '"ftrl", "momentum", "sgd" or "rmsprop".')

# Optimizer parameters specifc to adadelta
tf.app.flags.DEFINE_float(
  'adadelta_rho', 0.95,
  'The decay rate for adadelta.')

# Optimizer parameters specifc to adagrad
tf.app.flags.DEFINE_float(
  'adagrad_initial_accumulator_value', 0.1,
  'Starting value for the AdaGrad accumulators.')

# Optimizer parameters specifc to adam
tf.app.flags.DEFINE_float(
  'adam_beta1', 0.9,
  'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
  'adam_beta2', 0.999,
  'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

# Optimizer parameters specifc to ftrl
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
  'ftrl_initial_accumulator_value', 0.1,
  'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
  'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
  'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

# Optimizer parameters specifc to momentum
tf.app.flags.DEFINE_float(
  'momentum', 0.9,
  'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

# Optimizer parameters specifc to rmsprop_momentum
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# Logging parameters
tf.app.flags.DEFINE_integer(
  'stop_at_step', 1000, 'stop at stop')
tf.app.flags.DEFINE_integer(
  'log_loss_every_n_iters', 10, 'log_loss_every_n_iters')
tf.app.flags.DEFINE_integer(
  'profile_every_n_iters', 0, 'profile_every_n_iters')
tf.app.flags.DEFINE_integer(
  'profile_at_task', 0, 'profile_at_task')
tf.app.flags.DEFINE_bool(
  'log_device_placement', False,
  'Whether or not to log device placement.')
tf.app.flags.DEFINE_bool(
  'print_model_statistics', False, '')
tf.app.flags.DEFINE_string(
  'hooks', 'StopAtStepHook,ProfilerHook,LoggingTensorHook',
  'specify hooks for training.')

# Pipeline parameters
tf.app.flags.DEFINE_bool(
  'enable_pipeline', False, 'Whether run the model with pipelined model parallelism')
tf.app.flags.DEFINE_integer(
  'pipeline_device_num', 1, 'Number of devices used for pipeline running.')
tf.app.flags.DEFINE_integer(
  'pipeline_micro_batch_num', 1, 'Number of micro batches when running with pipeline.')
tf.app.flags.DEFINE_bool(
  'cross_pipeline', False, 'Whether run the model with pipelined across the nodes')

# Performance tuning parameters
tf.app.flags.DEFINE_bool('datasets_use_caching', False,
                         'Cache the compressed input data in memory. This improves '
                         'the data input performance, at the cost of additional '
                         'memory.')
tf.app.flags.DEFINE_string('protocol', 'grpc', 'default grpc. if rdma cluster, use grpc+verbs instead.')
tf.app.flags.DEFINE_string('variable_device_type', None, 'variable_device_type')
tf.app.flags.DEFINE_integer(
  'inter_op_parallelism_threads', 256, 'Compute pool size')
tf.app.flags.DEFINE_integer(
  'intra_op_parallelism_threads', 96, 'Eigen pool size')
tf.app.flags.DEFINE_bool(
  'use_grad_checkpoint', False,
  'Use gradient checkpoint to save gpu memory, default False')
tf.app.flags.DEFINE_integer(
  'labels_offset', 0,
  'An offset for the labels in the dataset. This flag is primarily used to '
  'evaluate the VGG and ResNet architectures which do not use a background '
  'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string('jitDeviceTuning', None, '')
tf.app.flags.DEFINE_string('deviceTuning', None, '')
tf.app.flags.DEFINE_string('deviceTuningTable', None, '')
tf.app.flags.DEFINE_string('jitDeviceTuningCacheHit', None, '')
tf.app.flags.DEFINE_bool('enable_bfloat16_sendrecv', False, '')

# Output parameter
tf.app.flags.DEFINE_string('tables', None, '')
tf.app.flags.DEFINE_string('buckets', None, '')
tf.app.flags.DEFINE_string('checkpointDir', None,
                           'The output directory where the model checkpoints will be written.')
tf.app.flags.DEFINE_integer("save_checkpoints_steps", 1000,
                            "How often to save the model checkpoint.")
tf.app.flags.DEFINE_string('summaryDir', None, '')
tf.app.flags.DEFINE_string('outputs', None, '')

###########
# For nmt #
###########
# network
tf.app.flags.DEFINE_integer('num_units', 32, 'network size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'network depth.')
tf.app.flags.DEFINE_integer('num_encoder_layers', None,
                            'encoder depth, equal to num_layers if None.')
tf.app.flags.DEFINE_integer('num_decoder_layers', None,
                            'decoder depth, equal to num_layers if None.')
tf.app.flags.DEFINE_string('encoder_type', 'bi',
                           'uni | bi | gnmt.'
                           'For bi, we build num_encoder_layers/2 bi-directional layers.'
                           'For gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1)'
                           'uni-directional layers.')
tf.app.flags.DEFINE_bool('residual', False, 'whether to add residual connections.')
tf.app.flags.DEFINE_bool('time_major', True,
                         'whether to use time-major mode for dynamic RNN.')
tf.app.flags.DEFINE_integer('num_embeddings_partitions', 0,
                            'number of partitions for embedding vars.')
# attention mechanisms
tf.app.flags.DEFINE_string('attention', '',
                           'luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no attention')
tf.app.flags.DEFINE_string('attention_architecture', 'standard',
                           'standard | gnmt | gnmt_v2.'
                           'standard: use top layer to compute attention.'
                           'gnmt: GNMT style of computing attention, use previous bottom layer to'
                           'compute attention.'
                           'gnmt_v2: similar to gnmt, but use current bottom layer to compute'
                           'attention.')
tf.app.flags.DEFINE_bool('output_attention', True,
                         'only used in standard attention_architecture. Whether use attention as '
                         'the cell output at each timestep.')
tf.app.flags.DEFINE_bool('pass_hidden_state', True,
                         'whether to pass hidden state of encoder to decoder when using an attention based model.')

# optimizer
tf.app.flags.DEFINE_integer('warmup_steps', 0, 'how many steps we inverse-decay learning.')
tf.app.flags.DEFINE_string('warmup_scheme', 't2t',
                           'how to warmup learning rates. Options include:'
                           't2t: Tensor2Tensor way, start with lr 100 times smaller,'
                           'then exponentiate until the specified lr.')
tf.app.flags.DEFINE_string('decay_scheme', '',
                           'How we decay learning rate. Options include:'
                           'luong234: after 2/3 num train steps, we start halving the learning rate'
                           'for 4 times before finishing.'
                           'luong5: after 1/2 num train steps, we start halving the learning rate'
                           'for 5 times before finishing.'
                           'luong10: after 1/2 num train steps, we start halving the learning rate'
                           'for 10 times before finishing.')
tf.app.flags.DEFINE_bool('colocate_gradients_with_ops', True,
                         'whether try colocating gradients with corresponding op')

# initializer
tf.app.flags.DEFINE_string('init_op', 'uniform', 'uniform | glorot_normal | glorot_uniform')
tf.app.flags.DEFINE_float('init_weight', 0.1,
                          'for uniform init_op, initialize weights between [-this, this].')
# data
tf.app.flags.DEFINE_string('src', None, 'source suffix, e.g., en.')
tf.app.flags.DEFINE_string('tgt', None, 'target suffix, e.g., de.')
tf.app.flags.DEFINE_string('train_prefix', None, 'train prefix, expect files with src/tgt suffixes.')
tf.app.flags.DEFINE_string('dev_prefix', None, 'dev prefix, expect files with src/tgt suffixes.')
tf.app.flags.DEFINE_string('test_prefix', None, 'test prefix, expect files with src/tgt suffixes.')
tf.app.flags.DEFINE_string('out_dir', None, 'store log/model files.')

# Vocab
tf.app.flags.DEFINE_string('vocab_prefix', None,
                           'vocab prefix, expect files with src/tgt suffixes.')
tf.app.flags.DEFINE_string('embed_prefix', None,
                           'Pretrained embedding prefix, expect files with src/tgt suffixes.'
                           'The embedding files should be Glove formated txt files.')
tf.app.flags.DEFINE_string('sos', '<s>', 'start-of-sentence symbol.')
tf.app.flags.DEFINE_string('eos', '</s>', 'end-of-sentence symbol.')
tf.app.flags.DEFINE_bool('share_vocab', False,
                         'whether to use the source vocab and embeddings for both source and target.')
tf.app.flags.DEFINE_bool('check_special_token', True,
                         'whether check special sos, eos, unk tokens exist in the vocab files.')

# Sequence lengths
tf.app.flags.DEFINE_integer('src_max_len', 50, 'max length of src sequences during training.')
tf.app.flags.DEFINE_integer('tgt_max_len', 50, 'max length of tgt sequences during training.')
tf.app.flags.DEFINE_integer('src_max_len_infer', None, 'max length of src sequences during inference.')
tf.app.flags.DEFINE_integer('tgt_max_len_infer', None, 'max length of tgt sequences during inference.')

# Default settings works well (rarely need to change)
tf.app.flags.DEFINE_string('unit_type', 'lstm', 'lstm | gru | layer_norm_lstm | nas')
tf.app.flags.DEFINE_float('forget_bias', 1.0, 'forget bias for BasicLSTMCell.')
tf.app.flags.DEFINE_float('dropout', 0.2, 'dropout rate (not keep_prob)')
tf.app.flags.DEFINE_float('max_gradient_norm', None, 'clip gradients to this norm.')
tf.app.flags.DEFINE_integer('max_train', 0,
                            'limit on the size of training data (0: no limit).')
tf.app.flags.DEFINE_integer('num_buckets', 5, 'put data into similar-length buckets.')
tf.app.flags.DEFINE_integer('num_sampled_softmax', 0,
                            'use sampled_softmax_loss if > 0. otherwise, use full softmax loss.')

# SPM
tf.app.flags.DEFINE_string('subword_option', '', 'set to bpe or spm to activate subword desegmentation.')

# Experimental encoding feature.
tf.app.flags.DEFINE_bool('use_char_encode', False,
                         'Whether to split each word or bpe into character, and then '
                         'generate the word-level representation from the character reprentation.')

# Misc
tf.app.flags.DEFINE_string('metrics', 'bleu', 'Comma-separated list of evaluations metrics (bleu,rouge,accuracy)')
tf.app.flags.DEFINE_integer('steps_per_external_eval', None,
                            'How many training steps to do per external evaluation.'
                            'Automatically set based on data if None.')
#tf.app.flags.DEFINE_string('scope', None, 'scope to put variables under')
tf.app.flags.DEFINE_string('hparams_path', None,
                           'path to standard hparams json file that overrides'
                           'hparams values from FLAGS.')
tf.app.flags.DEFINE_integer('random_seed', None, 'random seed (>0, set a specific seed).')
tf.app.flags.DEFINE_bool('override_loaded_hparams', False, 'override loaded hparams with values specified')
tf.app.flags.DEFINE_integer('num_keep_ckpts', 5, 'Max number of checkpoints to keep.')
tf.app.flags.DEFINE_bool('avg_ckpts', False,
                         'Average the last N checkpoints for external evaluation.'
                         'N can be controlled by setting --num_keep_ckpts.')
tf.app.flags.DEFINE_bool('language_model', False, 'True to train a language model, ignoring encoder')

# Inference
tf.app.flags.DEFINE_string('ckpt', '', 'Checkpoint file to load a model for inference.')
tf.app.flags.DEFINE_string('inference_input_file', None, 'Set to the text to decode.')
tf.app.flags.DEFINE_string('inference_list', None,
                           'A comma-separated list of sentence indices'
                           '(0-based) to decode.')
tf.app.flags.DEFINE_integer('infer_batch_size', 32, 'Batch size for inference mode.')
tf.app.flags.DEFINE_string('inference_output_file', None, 'Output file to store decoding results.')
tf.app.flags.DEFINE_string('inference_ref_file', None, 'Reference file to compute evaluation scores (if provided).')

# Advanced inference arguments
tf.app.flags.DEFINE_string('infer_mode', 'greedy',
                           'which type of decoder to use during inference. choices = [greedy, sample, beam_search]')
tf.app.flags.DEFINE_integer('beam_width', 0,
                            'beam width when using beam search decoder. If 0 (default), use standard'
                            'decoder with greedy helper.')
tf.app.flags.DEFINE_float('length_penalty_weight', 0.0, 'Length penalty for beam search.')
tf.app.flags.DEFINE_float('coverage_penalty_weight', 0.0, 'Coverage penalty for beam search.')
tf.app.flags.DEFINE_float('sampling_temperature', 0.0,
                          'Softmax sampling temperature for inference decoding, 0.0 means greedy'
                          'decoding. This option is ignored when using beam search.')
tf.app.flags.DEFINE_integer('num_translations_per_input', 1,
                            'Number of translations generated for each sentence. This is only used for'
                            'inference.')

tf.app.flags.DEFINE_bool('create_hparams', False, '')

# checkpoint/summary
tf.app.flags.DEFINE_string(
  "ckpt_file_name", None,
  "Initial checkpoint (pre-trained model: base_dir + model.ckpt).")
tf.app.flags.DEFINE_string(
  "model_config_file_name", None,
  "The config json file corresponding to the pre-trained model. "
  "This specifies the model architecture.")
tf.app.flags.DEFINE_string(
  "vocab_file_name", None,
  "The vocabulary file that the model was trained on.")

# misc
tf.app.flags.DEFINE_string("tensor_fusion_policy", 'default', '')
tf.app.flags.DEFINE_string("communication_policy", 'nccl_fullring', '')
tf.app.flags.DEFINE_integer("tensor_fusion_max_bytes", 32<<20, '')

FLAGS = tf.app.flags.FLAGS

