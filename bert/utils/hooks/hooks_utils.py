import tensorflow as tf


class LoggingTensorHook(tf.train.SessionRunHook):
  """Self-defined Hook for logging."""

  def __init__(self, tensors, samples_per_step=1, every_n_iters=100):
    self._tensors = tensors
    self._samples_per_step = samples_per_step
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_n_iters)

  def begin(self):
    self._timer.reset()
    self._tensors['global_step'] = tf.train.get_or_create_global_step()

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(self._tensors)

  def after_run(self, run_context, run_values):
    _ = run_context
    tensor_values = run_values.results
    stale_global_step = tensor_values['global_step']
    if self._timer.should_trigger_for_step(stale_global_step + 1):
      global_step = run_context.session.run(self._tensors['global_step'])
      if self._timer.should_trigger_for_step(global_step):
        elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
        if elapsed_time is not None:
          secs_per_step = '{0:.4f}'.format(elapsed_time / elapsed_steps)
          samples_per_sec = '{0:.4f}'.format(self._samples_per_step * elapsed_steps / elapsed_time)
          tf.logging.info("INFO:tensorflow:[%s secs/step,\t%s samples/sec]\t%s" % (
            secs_per_step, samples_per_sec,
            ',\t'.join(['%s = %s' % (tag, tensor_values[tag]) for tag in tensor_values])))
