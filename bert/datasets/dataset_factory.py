"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class dataset_factory(object):

  def __init__(self, module_name):
    self._module_name = module_name
    self._dataset_obj = None
    self._create_dataset_obj()

  def _create_dataset_obj(self):
    if not self._module_name:
      raise ValueError('Name of module is None.')
    module = __import__(self._module_name, globals(), locals(), [self._module_name])
    self._dataset_obj = getattr(module, self._module_name)

  def get_dataset_obj(self):
    return self._dataset_obj
