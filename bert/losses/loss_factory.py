"""A factory-pattern class which returns loss_fn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class loss_factory:

  def __init__(self, module_name):
    self._module_name = module_name
    self._loss_obj = None
    self._create_loss_obj()

  def _create_loss_obj(self):
    if not self._module_name:
      raise ValueError('Name of module is None.')
    module = __import__(self._module_name, globals(), locals(), [self._module_name])
    self._loss_obj = getattr(module, self._module_name)

  def get_loss_obj(self):
    return self._loss_obj
