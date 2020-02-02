#!/usr/bin/env python2
import tensorflow as tf
import numpy as np
import glob
import subprocess
import argparse
import os
import sys
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import applications
import horovod.tensorflow as hvd
from applications import cluster_utils

from tensorflow.python.ops import nccl_ops

import time
import pdb

flags = tf.app.flags

start = time.time()

def imgs_input_fn(filenames, labels=None, batch_size=64, fake_io=False):
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_image(image_string, channels=3)
        image.set_shape([None, None, None])
        global img_size
        global img_shape
        img_size = list(img_size)
        img_shape = list(img_shape)
        image = tf.image.resize_images(image, img_size)
        image.set_shape(img_shape)
        image = tf.reverse(image, axis=[2]) # 'RGB'->'BGR'
        d = image, label
        print('d=', d)
        return d
    def _parse_function_fake(filename, label):

        shape = tf.constant(np.array([[224, 224, 3]], dtype=np.int32))
        image = tf.constant(1.0, shape=[224, 224, 3])
        global img_size
        global img_shape
        print(img_shape, img_size)
        # img_size = list(img_size)
        # img_shape = list(img_shape)
        # image = tf.image.resize_images(image, img_size)
        # image.set_shape(shape)
        # image = tf.reverse(image, axis=[2]) # 'RGB'->'BGR'
        d = image, label
        return d
    if labels is None:
        labels = [0]*len(filenames)
    labels=np.array(labels)
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis=1)
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    labels = tf.cast(labels, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if fake_io is False:
        dataset = dataset.map(_parse_function)
    else:
        print('Using Fake IO')
        dataset = dataset.map(_parse_function_fake)
    dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(1)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size).prefetch(10)  # Batch size to use
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/replica:0/task:0/device:GPU:0'))
#    iterator = dataset.make_one_shot_iterator()
#    batch_features, batch_labels = iterator.get_next()
    # if using distribute.Strategy(), must use "return dataset" instead of "return batch_features, batch_lables"
    return dataset

def prepare_tf_config():
  hvd.init()
  ip_list = FLAGS.worker_hosts.split(',')
  worker_hosts = []
  for i, ip in enumerate(ip_list):
    worker_hosts.append(ip + ":" + str(4000 + i * 1000 + hvd.local_rank()))
  if len(worker_hosts) > 1:
    cluster = {"chief": [worker_hosts[0]],
               "worker": worker_hosts[1:]}
  else:
    cluster = {"chief": [worker_hosts[0]]}

  if FLAGS.task_index == 0:
    os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster,
         'task': {'type': "chief", 'index': 0}})
  else:
    os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster,
         'task': {'type': 'worker',
                  'index': FLAGS.task_index - 1}})


if __name__ == '__main__':
    default_raw_data_dir = '/tmp/dataset/mini-imagenet/raw-data/train/n01440764/'
    default_ckpt_dir = 'mycheckpoint'
    flags.DEFINE_string('model', 'resnet50', 'imagenet model name.')
    flags.DEFINE_string('strategy', 'none', 'strategy of variable updating')
    flags.DEFINE_integer('num_gpus', 1, 'number of GPUs used')
    flags.DEFINE_string('raw_data_dir', '', 'path to directory containing training dataset')
    flags.DEFINE_string('logdir', '', 'path to directory containing logs')
    flags.DEFINE_string('ckpt_dir', '', 'path to checkpoint directory')
    flags.DEFINE_integer('num_batches', 1, 'number of batches (a.k.a. steps or iterations')
    flags.DEFINE_integer('batch_size', 64, 'batch size per device (e.g. 32,64)')
    flags.DEFINE_string('worker', '', 'e.g. "host1:2222,host2:2222"')
    flags.DEFINE_string('ps', '', 'e.g. "host1:2220,host2:2220"')
    flags.DEFINE_string('task', '', 'e.g. "worker:0"')
    flags.DEFINE_bool('summary_only', False, '')
    flags.DEFINE_bool('fake_io', False, '')
    flags.DEFINE_string('mode', '', '')
    flags.DEFINE_string('worker_hosts', '', 'worker_hosts')
    flags.DEFINE_string('job_name', '', 'worker or ps')
    flags.DEFINE_integer('task_index', 0, '')
    flags.DEFINE_bool('cross_pipeline', False, '')
    flags.DEFINE_integer('pipeline_device_num', 1, '')
    flags.DEFINE_integer('micro_batch_num', 1, '')
    flags.DEFINE_integer('num_replica', 1, '')
    flags.DEFINE_bool('fp16', False, 'whether enable fp16 communication while transfer activations between stages')

    FLAGS = flags.FLAGS

    os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'

    subprocess.call('rm -rf /tmp/tmp*',shell=True)
    print("logdir = %s" % FLAGS.logdir)
    if os.path.isdir(FLAGS.logdir):
        subprocess.call('rm -rf %s/*' % FLAGS.logdir, shell=True)
    else:
        subprocess.call('mkdir -p %s' % FLAGS.logdir, shell=True)

    if FLAGS.model is not None:
        print('Training model: ', FLAGS.model)
        if FLAGS.model == 'vgg19':
            model = applications.vgg19.SliceVGG19(weights=None)
        else:
            print("No model")
            exit(1)
    if FLAGS.summary_only:
        print(model.summary())
        sys.exit(0)

    subprocess.call('rm -rf %s' % FLAGS.ckpt_dir, shell=True)
    global img_shape
    global img_size
#    shape = list(model.layers[0].output_shape)
#    shape.pop(0)
    img_shape = (224, 224, 3)
#    shape.pop(-1)
    img_size = (224, 224)

    tf.logging.set_verbosity(tf.logging.DEBUG)

    if FLAGS.fake_io is False:
        filenames = glob.glob(FLAGS.raw_data_dir+"*.JPEG")
    else:
        filenames = ["biu.JPEG"] * 1048576
    if FLAGS.strategy == 'none':
        distribution = None
    else:
        print("Error distribute strategy!")
        exit(1)

    prepare_tf_config()
    session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(allow_growth=True))
    if FLAGS.cross_pipeline:
      cluster_manager = cluster_utils.get_cluster_manager(config_proto=session_config)
    config = tf.estimator.RunConfig(
            train_distribute = distribution,
            log_step_count_steps=10,
            save_checkpoints_secs=None,
            save_checkpoints_steps=None, ## QQ: disable checkpoints
            session_config=session_config)

    def get_train_op(grads_and_vars, slice_num=None, replica_length=None, vars_org=None):
      global_step = tf.train.get_or_create_global_step()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
      gradients, variables = zip(*grads_and_vars)
      if FLAGS.cross_pipeline and hvd.size() > 1:
        # pipeline num device
        devices = cluster_utils.get_pipeline_devices(FLAGS.pipeline_device_num)
        if FLAGS.num_replica >= 2:
          replica_devices = cluster_utils.get_replica_devices(FLAGS.num_replica)
          replica_grads_list = [[] for i in xrange(len(replica_devices))]
          for grad, var in grads_and_vars:
            for i in xrange(len(replica_devices)):
              if var.device == replica_devices[i]:
                replica_grads_list[i].append((grad, var))
                break
          replica_avg_grads_and_vars = []
          for i in xrange(len(replica_devices)):
            with tf.device(replica_devices[i]):
              for grad, var in replica_grads_list[i]:
                if isinstance(grad, tf.IndexedSlices):
                  grad = tf.convert_to_tensor(grad)
                avg_grad = hvd.allreduce(grad)
                avg_grads_and_vars.append((avg_grad, var))
          grads_and_vars = avg_grads_and_vars
          # Note:
          # As stage0:stage1=15:1, so stage1 only holds one device
          # and no DataParallel is applied on top of DAPPLE unit,
          # as a result no hvd.allreduce() of weights of stage1 is applied here.
        else:
          gradients_list = [[] for i in xrange(len(devices))]
          for grad, var in grads_and_vars:
            for i in xrange(len(devices)):
              if var.device == devices[i]:
                gradients_list[i].append((grad, var))
                break
          avg_grads_and_vars = []
          for i in xrange(len(devices)):
            with tf.device(devices[i]):
              for grad, var in gradients_list[i]:
                if isinstance(grad, tf.IndexedSlices):
                  grad = tf.convert_to_tensor(grad)
                avg_grad = hvd.allreduce(grad)
                avg_grads_and_vars.append((avg_grad, var))
          grads_and_vars = avg_grads_and_vars

      train_op = optimizer.apply_gradients(
          grads_and_vars, global_step=global_step)
############ fsq175703, FIXME later
      #if FLAGS.num_replica > 1:
      #  if vars_org is not None:
      #    if slice_num == 2:
      #      length = replica_length
      #      offset = length / FLAGS.num_replica
      #      update = []
      #      with tf.control_dependencies([train_op]):
      #        for i in xrange(1, FLAGS.num_replica):
      #          for j in xrange(offset):
      #            with tf.device(vars_org[i*offset + j].device):
      #              update.append(vars_org[i*offset + j].assign(vars_org[j] if i < split_gpu+1 else vars_org[split_gpu*offset + j]))
      #        train_op = tf.group(update)
      #    else:
      #      print("replica weights update error!")
      #      exit(1)
      #  else:
      #    print("replica weights update error!")
      #    exit(1)
      return train_op

    def model_fn(features, labels, mode):
      total_loss, outputs = model.build(features, labels)
      if FLAGS.pipeline_device_num <= 1:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        grads_and_vars = optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
      else:
#######################
        devices = cluster_utils.get_pipeline_devices(FLAGS.pipeline_device_num)
        slice_num = len(devices)
        micro_batch_num = FLAGS.micro_batch_num
        losses = []
        all_outputs = []
        losses.append(total_loss)
        all_outputs.append(outputs)
        layer_grads = [[[] for i in xrange(slice_num)] for j in xrange(micro_batch_num)]
        layer_vars = [[] for i in xrange(slice_num)]
        remained_vars = tf.trainable_variables()
        ys = losses[0]
        prev_grads=None
        # layers-1 ~ 1 compute grads
        for i in xrange(slice_num - 1, 0, -1):
          vars_i = [v for v in remained_vars if v.device==devices[i]]
          remained_vars = [v for v in remained_vars if v not in vars_i]
          prev_y = all_outputs[0][i-1]
          y_grads = tf.gradients(ys=ys, xs=[prev_y]+vars_i, grad_ys=prev_grads, colocate_gradients_with_ops=True)
          ys = prev_y
          prev_grads = y_grads[0]
          grads_i = y_grads[1:]
          layer_grads[0][i] = [g for g in grads_i if g is not None]
          layer_vars[i] = [v for (g, v) in zip(grads_i, vars_i) if g is not None]
        # layer 0 compute grads
        grads_0 = tf.gradients(ys=ys, xs=remained_vars, grad_ys=prev_grads, colocate_gradients_with_ops=True)
        layer_grads[0][0] = [g for g in grads_0 if g is not None]
        layer_vars[0] = [v for (g, v) in zip(grads_0, remained_vars) if g is not None]

        # other micro_batch_num
        for j in xrange(1, micro_batch_num):
          dep_outputs = []
          for i in xrange(slice_num):
            dep_outputs.append(all_outputs[j-1][i] if i+j < 2*slice_num-1 else layer_grads[i+j-2*slice_num+1][i])
            # dep_outputs.append(all_outputs[j-1][i] if i+j < slice_num else layer_grads[i+j-slice_num][i])
          loss, outputs = model.build(features, labels, dep_outputs=dep_outputs)
          losses.append(loss)
          all_outputs.append(outputs)
          ys = losses[j]
          prev_grads=None
          for i in xrange(slice_num - 1, 0, -1):
            prev_y = all_outputs[j][i-1]
            y_grads = tf.gradients(ys=ys, xs=[prev_y]+layer_vars[i], grad_ys=prev_grads, colocate_gradients_with_ops=True)
            ys = prev_y
            prev_grads = y_grads[0]
            grads_i = y_grads[1:]
            layer_grads[j][i] = [g for g in grads_i if g is not None]
          grads_0 = tf.gradients(ys=ys, xs=layer_vars[0], grad_ys=prev_grads, colocate_gradients_with_ops=True)
          layer_grads[j][0] = [g for g in grads_0 if g is not None]

        grads_set_org = []
        vars_set_org = []
        for i in xrange(slice_num):
          for j in xrange(len(layer_grads[0][i])):
            grad_i_set = [layer_grads[m][i][j] for m in range(micro_batch_num)]
            #print (grad_i_set)
            if micro_batch_num == 1:
              with tf.device(grad_i_set[0].device):
                acc_grads = grad_i_set[0]
            else:
              with tf.control_dependencies(grad_i_set), tf.device(grad_i_set[0].device): # replica
                if isinstance(grad_i_set[0], tf.IndexedSlices):
                  acc_grads = tf.add_n(grad_i_set)
                else:
                  acc_grads = tf.accumulate_n(grad_i_set)
            grads_set_org.append(acc_grads)
            vars_set_org.append(layer_vars[i][j])
        if FLAGS.num_replica > 1:
################
          # Note the grads_set_org looks like the following:
          #  (grad0_gpu0, grad1_gpu0, grad2_gpu0, ..., gradN_gpu0,
          #   grad0_gpu1, grad1_gpu1, grad2_gpu1, ..., gradN_gpu1,
          #   ...
          #   grad0_gpuR, grad1_gpuR, grad2_gpuR, ..., gradN_gpuR)
          if slice_num ==2:
            new_grads_set = []
            new_vars_set = []
            length = len(layer_grads[0][0])
            offset = length / FLAGS.num_replica

            ## Option1: use hvd.allreduce
            avg_grads_and_vars = []
            for i in xrange(offset):
              for j in xrange(FLAGS.num_replica):
                grad = grads_set_org[j*offset + i]
                with tf.device(grad.device):
                  avg_grad = hvd.allreduce(grad)
                avg_grads_and_vars.append((avg_grad, vars_set_org[j*offset+i]))
            for i in xrange(length, len(grads_set_org)):
              avg_grads_and_vars.append((grads_set_org[i], vars_set_org[i]))
            grads_and_vars = avg_grads_and_vars

            #for i in xrange(offset):
            #  grads_i = []
            #  for j in xrange(FLAGS.num_replica):
            #    grads_i.append(grads_set_org[j*offset + i])
            #  global split_gpu
            #  split_gpu = 2
            #  summed_grads_i_0 = nccl_ops.all_sum(grads_i[:split_gpu])[0] ### 0~7 device on machine0
            #  summed_grads_i_1 = nccl_ops.all_sum(grads_i[split_gpu:])[0] ### 0~6 device on machine1
            #  with tf.device(summed_grads_i_0.device):
            #    summed_grads_i_0 += summed_grads_i_1 ## Cross machine communication!
            #  new_grads_set.append(summed_grads_i_0)
            #  new_vars_set.append(vars_set_org[i])
            #grads_set = new_grads_set + grads_set_org[length:]
            #vars_set = new_vars_set + vars_set_org[length:]
            #grads_and_vars = zip(grads_set, vars_set)
################
          #if slice_num == 2:
          #  length = len(layer_grads[0][0])
          #  offset = length / FLAGS.num_replica
          #  for i in xrange(1,FLAGS.num_replica):
          #    for j in xrange(offset):
          #      grads_set_org[j] += grads_set_org[i*offset + j]
          #  grads_set = grads_set_org[0:offset] + grads_set_org[length:]
          #  vars_set = vars_set_org[0:offset] + vars_set_org[length:]
          #grads_and_vars = zip(grads_set, vars_set)
          else:
            print("replica gradients aggregation error!")
            exit(1)
        else:
          grads_set = grads_set_org
          vars_set = vars_set_org
          grads_and_vars = zip(grads_set, vars_set)
#######################
      train_op = get_train_op(grads_and_vars, slice_num, length, vars_set_org)
      return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op, )
    est = tf.estimator.Estimator(model_fn=model_fn, config=config)
   # Create a callalble object: fn_image_preprocess
    fn_image_preprocess = lambda : imgs_input_fn(filenames, None, FLAGS.batch_size, FLAGS.fake_io)
    print("Training starts.")
    start = time.time()
    est.train( input_fn = fn_image_preprocess , steps = FLAGS.num_batches) #, hooks = hooks)
    print('Elapsed ', time.time() - start, 's for ', FLAGS.num_batches, ' batches')
