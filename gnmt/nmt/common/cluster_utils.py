import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.python.platform import flags
import pdb

FLAGS = None
MAX_GPUS_PER_NODE = 8

def get_cluster_manager(flags, config_proto):
  """Returns the cluster manager to be used."""
  global FLAGS
  FLAGS = flags
  return GrpcClusterManager(config_proto)

def get_pipeline_devices(num_devices):
  # Not pipeline, used for baseline test on single GPU.
  if not FLAGS:
    return ['/gpu:0', '/gpu:0']
    #return ['/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:0']
  # Pipeline case
  devices = []
  if FLAGS.job_name == "":
    FLAGS.job_name = "localhost"
  for i in xrange(num_devices):
    gpu_id = hvd.local_rank() * FLAGS.num_replica
    if FLAGS.num_replica > MAX_GPUS_PER_NODE:
      if i == 1:
        gpu_id = FLAGS.num_replica % MAX_GPUS_PER_NODE
    
    devices.append("/job:%s/replica:0/task:%d/device:GPU:%d" % 
                   (FLAGS.job_name, i, gpu_id))
  print("QQ: devices:")
  print(devices)
  return devices

def get_replica_devices(num_replica):
    devices = []
    for i in xrange(num_replica):
      devices.append("/job:%s/replica:0/task:%d/device:GPU:%d" % (FLAGS.job_name, \
                                             i / MAX_GPUS_PER_NODE, \
                                             i % MAX_GPUS_PER_NODE))
    return devices

def get_inner_replica_devices(num_replica):
    devices = []
    for i in xrange(num_replica):
      devices.append("/gpu:%d" % (hvd.local_rank() * num_replica + i))
    return devices


class BaseClusterManager(object):
  """The manager for the cluster of servers running the fast-nn."""

  def __init__(self):
    assert FLAGS.job_name in ['worker', None, ''], 'job_name must be worker'
    if FLAGS.job_name and FLAGS.worker_hosts:
      ip_list = FLAGS.worker_hosts.split(',')
      worker_hosts = []
      for i, ip in enumerate(ip_list):
        worker_hosts.append(ip + ":" + str(4323 + i * 1000 + hvd.local_rank()))
      print(worker_hosts)
      cluster_dict = {'worker': worker_hosts}
    else:
      cluster_dict = {'worker': ['127.0.0.1:0']}

    self._num_workers = len(cluster_dict['worker'])
    self._cluster_spec = tf.train.ClusterSpec(cluster_dict)
    self._device_exp = tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d/" % FLAGS.task_index,
      cluster=self._cluster_spec)

  def get_target(self):
    """Returns a target to be passed to tf.Session()."""
    raise NotImplementedError('get_target must be implemented by subclass')

  def get_cluster_spec(self):
    return self._cluster_spec

  def num_workers(self):
    return self._num_workers

  def device_exp(self):
    return self._device_exp


class GrpcClusterManager(BaseClusterManager):
  """A cluster manager for a cluster networked with gRPC."""

  def __init__(self, config_proto):
    super(GrpcClusterManager, self).__init__()
    self._server = tf.train.Server(self._cluster_spec,
                                   job_name=FLAGS.job_name,
                                   task_index=FLAGS.task_index,
                                   config=config_proto,
                                   protocol='grpc')
    # hang the non-chief workers
    if FLAGS.cross_pipeline:
      if FLAGS.task_index != 0:
        self._server.join()

    self._target = self._server.target

  def get_target(self):
    return self._target
