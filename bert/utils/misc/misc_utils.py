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

"""Generally useful utility functions."""
from __future__ import print_function

import codecs
import json
import math
import os
import sys
import time
from distutils import version

import six
import tensorflow as tf


def check_tensorflow_version():
  # LINT.IfChange
  min_tf_version = "1.12.0"
  # LINT.ThenChange(<pwd>/nmt/copy.bara.sky)
  if (version.LooseVersion(tf.__version__) <
      version.LooseVersion(min_tf_version)):
    raise EnvironmentError("Tensorflow version must >= %s" % min_tf_version)


def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans


def print_time(s, start_time):
  """Take a start time, print elapsed duration, and return a new time."""
  print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
  sys.stdout.flush()
  return time.time()


def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  if six.PY2:
    sys.stdout.write(s.encode("utf-8"))
  else:
    sys.stdout.buffer.write(s.encode("utf-8"))

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()


def print_hparams(hparams, skip_patterns=None, header=None):
  """Print hparams, can skip keys based on pattern."""
  if header: print_out("%s" % header)
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print_out("  %s=%s" % (key, str(values[key])))


def load_hparams(model_dir):
  """Load hparams from an existing model directory."""
  hparams_file = os.path.join(model_dir, "hparams")
  if tf.gfile.Exists(hparams_file):
    print_out("# Loading hparams from %s" % hparams_file)
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
      try:
        hparams_values = json.load(f)
        hparams = tf.contrib.training.HParams(**hparams_values)
      except ValueError:
        print_out("  can't load hparams file")
        return None
    return hparams
  else:
    return None


def maybe_parse_standard_hparams(hparams, hparams_path):
  """Override hparams values with existing standard hparams config."""
  if hparams_path and tf.gfile.Exists(hparams_path):
    print_out("# Loading standard hparams from %s" % hparams_path)
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_path, "rb")) as f:
      hparams.parse_json(f.read())
  return hparams


def save_hparams(out_dir, hparams):
  """Save hparams."""
  hparams_file = os.path.join(out_dir, "hparams")
  print_out("  saving hparams to %s" % hparams_file)
  with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
    f.write(hparams.to_json(indent=4, sort_keys=True))


def debug_tensor(s, msg=None, summarize=10):
  """Print the shape and value of a tensor at test time. Return a new tensor."""
  if not msg:
    msg = s.name
  return tf.Print(s, [tf.shape(s), s], msg + " ", summarize=summarize)




