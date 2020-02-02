# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import optimization
import tensorflow as tf
from tensorflow import logging
import tokenization
import util
import os
from tensorflow.python.framework import ops

from tensorflow.contrib import nccl

dim = 10000

with tf.device('/gpu:0'):
    a = tf.get_variable(
            "a", initializer=tf.constant(1.0, shape=(dim, dim)))

with tf.device('/gpu:1'):
    b = tf.get_variable(
            "b", initializer=tf.constant(2.0, shape=(dim, dim)))

with tf.device('/gpu:0'):
    summed_node = nccl.all_sum([a, b])
    for i in summed_node:
        print('before', i, i.device)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

with tf.device('/gpu:0'):
    summed = sess.run(summed_node)
    #print('summed: ', summed)

