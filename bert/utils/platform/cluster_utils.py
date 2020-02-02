import tensorflow as tf
from train_flags import FLAGS
import horovod.tensorflow as hvd


def get_cluster_manager(config_proto):
  """Returns the cluster manager to be used."""
  return GrpcClusterManager(config_proto)


class BaseClusterManager(object):
  """The manager for the cluster of servers running the fast-nn."""

  def __init__(self):
    assert FLAGS.job_name in ['worker', None, ''], 'job_name must be worker'
    if FLAGS.job_name and FLAGS.worker_hosts:
      ip_list = FLAGS.worker_hosts.split(',')
      worker_hosts = []
      for i, ip in enumerate(ip_list):
        worker_hosts.append(ip + ":" + str(4000 + i * 1000 + hvd.local_rank()))
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
                                   protocol=FLAGS.protocol)
    # hang the non-chief workers
    if FLAGS.cross_pipeline:
      if FLAGS.task_index != 0:
        self._server.join()

    self._target = self._server.target

  def get_target(self):
    return self._target
