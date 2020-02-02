import tensorflow as tf


class BaseLoss(object):

  def __init__(self, dataset_iterator, network_fn, hparams=None):
    self._dataset_iterator = dataset_iterator
    self._network_fn = network_fn
    self._hparams = hparams
    self._loss = None
    self._mean_accuracy = []
    self._stages = self.stages
    self._loss_fn = self.loss_fn

  def _get_accuracy(self):
    return self._mean_accuracy

  def _get_loss_fn(self):
    return self._loss_fn

  def _get_stages(self):
    return self._stages

  def loss_fn(self):
    """Function to specific loss_fn, userParams should specify the 'loss_name', default 'example.py'."""
    raise NotImplementedError('Must be implemented by subclass.')
