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

import time
import pdb
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
  worker_hosts = args.worker_hosts.split(',')
  if len(worker_hosts) > 1:
    cluster = {"chief": [worker_hosts[0]],
               "worker": worker_hosts[1:]}
  else:
    cluster = {"chief": [worker_hosts[0]]}

  # Horovod Debug
  os.environ['HOROVOD_HIERARCHICAL_ALLREDUCE'] = '1'
  os.environ['HOROVOD_LOG_LEVEL'] = 'DEBUG'

  if args.task_index == 0:
    os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster,
         'task': {'type': "chief", 'index': 0}})
  else:
    os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster,
         'task': {'type': 'worker',
                  'index': args.task_index - 1}})
  os.environ['NCCL_DEBUG'] = 'INFO'

def make_distributed_info_without_evaluator():
  worker_hosts = args.worker_hosts.split(",")
  if len(worker_hosts) > 1:
    cluster = {"chief": [worker_hosts[0]],
               "worker": worker_hosts[1:]}
  else:
    cluster = {"chief": [worker_hosts[0]]}

  os.environ['HOROVOD_HIERARCHICAL_ALLREDUCE'] = '1'
  os.environ['HOROVOD_LOG_LEVEL'] = 'DEBUG'

  if args.task_index == 0:
    task_type = "chief"
    task_index = 0
  else:
    task_type = "worker"
    task_index = args.task_index - 1
  return cluster, task_type, task_index

def dump_into_tf_config(cluster, task_type, task_index):
  os.environ['TF_CONFIG'] = json.dumps(
      {'cluster': cluster,
       'task': {'type': task_type, 'index': task_index}})

if __name__ == '__main__':
    default_raw_data_dir = '/tmp/dataset/mini-imagenet/raw-data/train/n01440764/'
    default_ckpt_dir = 'mycheckpoint'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='imagenet model name.', default='resnet50')
    parser.add_argument('--strategy', help='strategy of variable updating', default='none')
    parser.add_argument('--num_gpus', help='number of GPUs used', default=1, type=int)
    parser.add_argument('--raw_data_dir', help='path to directory containing training dataset', default=default_raw_data_dir)
    parser.add_argument('--logdir', help='path to directory containing logs', default='train_log')
    parser.add_argument('--ckpt_dir', help='path to checkpoint directory', default=default_ckpt_dir)
    parser.add_argument('--num_batches', help='number of batches (a.k.a. steps or iterations', type=int, default=10)
    parser.add_argument('--batch_size', help='batch size per device (e.g. 32,64)', type=int, default=64)
    # parser.add_argument('--worker', help='e.g. "host1:2222,host2:2222"')
    parser.add_argument('--ps', help='e.g. "host1:2220,host2:2220"')
    parser.add_argument('--task', help='e.g. "worker:0"')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--fake_io', type=bool, default=False)
    parser.add_argument('--mode')
    parser.add_argument('--worker_hosts')
    parser.add_argument('--job_name')
    parser.add_argument('--task_index', type=int)
    parser.add_argument('--cross_pipeline', type=bool, default=False)
    parser.add_argument('--pipeline_device_num', type=int, default=1)

    args = parser.parse_args()

    
    os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'

    subprocess.call('rm -rf /tmp/tmp*',shell=True)
    print("logdir = %s" % args.logdir)
    if os.path.isdir(args.logdir):
        subprocess.call('rm -rf %s/*' % args.logdir, shell=True) 
    else:
        subprocess.call('mkdir -p %s' % args.logdir, shell=True) 

    if args.model is not None:
        print('Training model: ', args.model)
        if args.model == 'vgg19':
            model = applications.vgg19.VGG19(weights=None)
        else:
            print("No model")
            exit(1)
    if args.summary_only:
        print(model.summary())
        sys.exit(0)
  
    subprocess.call('rm -rf %s' % args.ckpt_dir, shell=True)
    global img_shape
    global img_size
#    shape = list(model.layers[0].output_shape)
#    shape.pop(0)
    img_shape = (224, 224, 3)
#    shape.pop(-1)
    img_size = (224, 224)

    tf.logging.set_verbosity(tf.logging.DEBUG)

    if args.fake_io is False:
        filenames = glob.glob(args.raw_data_dir+"*.JPEG")
    else: 
        filenames = ["biu.JPEG"] * 1048576
    # if using distribute.Strategy(), only tensorflow native optimizer is allowed currently.
    if args.strategy == 'mirrored':
        distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus)
    elif args.strategy == 'parameter_server':
        distribution = tf.contrib.distribute.ParameterServerStrategy()
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7' 
    elif args.strategy == 'collective':
        cluster, task_type, task_index = make_distributed_info_without_evaluator()
        dump_into_tf_config(cluster, task_type, task_index)
        # if args.mode == "pai":
        #     prepare_tf_config()
        distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(num_gpus_per_worker=args.num_gpus, cross_tower_ops_type='horovod',
        all_dense=True)
    elif args.strategy == 'none':
        distribution = None

    config = tf.estimator.RunConfig( train_distribute = distribution, log_step_count_steps=100)
    def get_train_op(grads_and_vars):
      global_step = tf.train.get_or_create_global_step()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
      gradients, variables = zip(*grads_and_vars)
      if args.cross_pipeline and hvd.size() > 1:
        # pipeline num device
        devices = cluster_utils.get_pipeline_devices(args.pipeline_device_num)
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
      return train_op

    def model_fn(features, labels, mode):
      loss = model.build(features, labels)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
      grads_and_vars = optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
      train_op = get_train_op(grads_and_vars)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, )
    est = tf.estimator.Estimator(model_fn=model_fn, config=config)
   # Create a callalble object: fn_image_preprocess
    fn_image_preprocess = lambda : imgs_input_fn(filenames, None, args.batch_size, args.fake_io)
    print("Training starts.")
    start = time.time()
    est.train( input_fn = fn_image_preprocess , steps = args.num_batches) #, hooks = hooks)
    print('Elapsed ', time.time() - start, 's for ', args.num_batches, ' batches')
