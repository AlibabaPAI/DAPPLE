"""Hooks helper to return a list of TensorFlow hooks for training by name.

More hooks can be added to this set. To add a new hook,
1) add the new hook to the registry in HOOKS,
2) add a corresponding function that parses out necessary parameters.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from utils.hooks import hooks_utils
from train_flags import FLAGS


def get_train_hooks(params, **kwargs):
  """Factory for getting a list of TensorFlow hooks for training by name.

  Args:
      name_list: a list of strings to name desired hook classes. Allowed:
      StopAtStepHook, ProfilerHook, LoggingTensorHook, which are defined
      as keys in HOOKS
      tensors: dict of tensor names to be logged.
      **kwargs: a dictionary of arguments to the hooks.

    Returns:
      list of instantiated hooks, ready to be used in a classifier.train call.

    Raises:
      ValueError: if an unrecognized name is passed.
    """
  if not FLAGS.hooks:
    return []

  name_list = FLAGS.hooks.split(',')
  train_hooks = []
  for name in name_list:
    hook_name = HOOKS.get(name.strip().lower())
    if hook_name is None:
      raise ValueError('Unrecognized training hook requested: {}'.format(name))
    else:
      res = hook_name(params, **kwargs)
      if res:
        train_hooks.append(res)

  return train_hooks


def get_logging_tensor_hook(params, **kwargs):
  """Function to get LoggingTensorHook.
  Args:
      tensors: dict of tensor names.
      samples_per_step:num of samples that machines process at one step.
      every_n_iter: `int`, print the values of `tensors` once every N local steps taken on the current worker.
      **kwargs: a dictionary of arguments to LoggingTensorHook.

    Returns:
      Returns a LoggingTensorHook with a standard set of tensors that will be
      printed to stdout or Null.
  """
  if FLAGS.log_loss_every_n_iters > 0:
    return hooks_utils.LoggingTensorHook(
      tensors=params['tensors_to_log'],
      samples_per_step=params['samples_per_step'],
      every_n_iters=FLAGS.log_loss_every_n_iters)
  else:
    pass


def get_profiler_hook(params, **kwargs):
  """Function to get ProfilerHook.
  Args:
      model_dir: The directory to save the profile traces to.
      save_steps: `int`, print profile traces every N steps.
      **kwargs: a dictionary of arguments to ProfilerHook.

  Returns:
      Returns a ProfilerHook that writes out timelines that can be loaded into
      profiling tools like chrome://tracing or Null.
  """
  if FLAGS.profile_every_n_iters > 0 and FLAGS.task_index == FLAGS.profile_at_task:
    return tf.train.ProfilerHook(
      save_steps=FLAGS.profile_every_n_iters,
      output_dir='.')
  else:
    pass


def get_stop_at_step_hook(params, **kwargs):
  """Function to get StopAtStepHook.

  Args:
      last_step: `int`, training stops at N steps.
      **kwargs: a dictionary of arguments to StopAtStepHook.

  Returns:
      Returns a StopAtStepHook.
  """
  return tf.train.StopAtStepHook(last_step=FLAGS.stop_at_step)


# A dictionary to map one hook name and its corresponding function
HOOKS = {
  'loggingtensorhook': get_logging_tensor_hook,
  'profilerhook': get_profiler_hook,
  'stopatstephook': get_stop_at_step_hook,
}
