import tensorflow as tf

import numpy as np
import pdb

flags = tf.app.flags
# Optional value of model:
#   amoeba, bert, xlnet, vgg19, resnet50 and gnmt16
flags.DEFINE_string('model', 'gnmt16', 'worker or ps')
## server config
# p3.2xlarge 1V100, up to 10Gpbs
# p3.8xlarge 4V100, 10Gpbs
# p3.16xlarge 8V100, 25Gbps
flags.DEFINE_string('server', 'p3.16xlarge', 'server config')
flags.DEFINE_bool('only_pipe', False, 'straight pipe')
flags.DEFINE_bool('dp_best_overlap', True,
                  'assume data parallel can over allreduce communication best')
flags.DEFINE_bool('pipe_dream_cut', False, 'pipe dream')
flags.DEFINE_bool('DGX_2H', False, 'pipe dream')
flags.DEFINE_bool('verbose', False, 'verbose log')
flags.DEFINE_integer('total_gpu_num', 16, 'total num of gpus')
flags.DEFINE_integer('total_batch_size', 4096, 'total batch size')
FLAGS = flags.FLAGS

print("### FLAGS.server=%s ###" % FLAGS.server)
if FLAGS.server == "pipe-torch":
  # https://ieeexplore.ieee.org/document/8916305
  MAX_GPUS_PER_NODE = 2
elif FLAGS.server == "p3.2xlarge":
  MAX_GPUS_PER_NODE = 1
elif FLAGS.server == "p3.8xlarge":
  MAX_GPUS_PER_NODE = 4
elif FLAGS.server == "p3.16xlarge":
  MAX_GPUS_PER_NODE = 8
elif FLAGS.DGX_2H:
  MAX_GPUS_PER_NODE = 16

def allreduce_vol(weights, ring_len):
  # MB
  return 2 * float(ring_len - 1) / float(ring_len) * float(weights)

class VGG19(object):
  def __init__(self):
   # model setting
   # time:ms size:MB
   self.max_batch_size = 190
   self.min_batch_size = 8
   self.weights = 548.0
   self.fw_time = 50.0
   self.bw_time = 100.0
   self.fw_bw_time = self.fw_time + self.bw_time
   self.profile_batch_size = 32
   self.weights_of_last_layer = 65
   # stage 1 for min_batch_size
   self.cut_features = 0.8 * self.min_batch_size
   if FLAGS.pipe_dream_cut and FLAGS.total_gpu_num == 16:
     self.cut_features = 0.2 * self.min_batch_size
     self.comp_cut = [9.8, 3.0] # 147, 3
     self.devices_cut = [15, 1]
     if FLAGS.total_gpu_num == 8:
       self.comp_cut = [6.0, 6.0] # 42, 6
       self.devices_cut = [7, 1]
     self.weights_cut = [76, 471.6]
   else:
     self.cut_features = 0.8 * self.min_batch_size
     self.comp_cut = [50.0 / 4, 50.0 / 4] # 150, 50
     self.devices_cut = [2, 1]
     self.weights_cut = [8.7, 539.3]

  def compute_time(self, batch_size):
    # ms
    scaled_fw_bw_time = self.fw_bw_time / self.profile_batch_size * float(batch_size)
    scaled_bw_time = self.bw_time / self.profile_batch_size * \
        (batch_size if batch_size <= self.max_batch_size else self.max_batch_size)
    return scaled_fw_bw_time, scaled_bw_time
 
class XLNet(object):
  def __init__(self):
   # model setting
   # time:ms size:MB
   self.max_batch_size = 2
   self.min_batch_size = 1
   self.weights = 1460.0
   self.fw_time = 80.0
   self.bw_time = 140.0
   self.fw_bw_time = self.fw_time + self.bw_time
   self.profile_batch_size = 1
   self.weights_of_last_layer = 52
   # stage 1 for min_batch_size
   self.cut_features = 10 * self.min_batch_size
   if FLAGS.only_pipe:
     assert FLAGS.total_gpu_num == 16
     self.comp_cut = [13.75] * FLAGS.total_gpu_num
     self.devices_cut = [1] * FLAGS.total_gpu_num
     self.weights_cut = [131] + [88] * (FLAGS.total_gpu_num - 1)
   else:
     stages = FLAGS.total_gpu_num / MAX_GPUS_PER_NODE
     scale = stages / 2
     self.comp_cut = [110 / scale, 110 / scale] * scale
     self.devices_cut = [1, 1] * scale 
     self.weights_cut = [690 / scale, 770 / scale] * scale 

  def compute_time(self, batch_size):
    # ms
    scaled_fw_bw_time = self.fw_bw_time / self.profile_batch_size * float(batch_size)
    scaled_bw_time = self.bw_time / self.profile_batch_size * \
        (batch_size if batch_size <= self.max_batch_size else self.max_batch_size)
    return scaled_fw_bw_time, scaled_bw_time

class BertLarge(object):
  def __init__(self):
   # model setting
   # time:ms size:MB
   self.max_batch_size = 6
   self.min_batch_size = 2
   self.weights = 1360.0
   self.fw_time = 77
   self.bw_time = 129
   self.fw_bw_time = self.fw_time + self.bw_time
   self.profile_batch_size = 2.0
   # The last layer of bertLarge is pooler layer
   self.weights_of_last_layer = 48
   # stage 1 for min_batch_size
   self.cut_features = 1.5 * self.min_batch_size
   stages = FLAGS.total_gpu_num / MAX_GPUS_PER_NODE
   scale = stages / 2
   self.comp_cut = [103.0 / scale, 103.0 / scale] * scale 
   self.devices_cut = [1, 1] * scale
   self.weights_cut = [660 / scale, 700 / scale] * scale 

  def compute_time(self, batch_size):
    # ms
    scaled_fw_bw_time = self.fw_bw_time / self.profile_batch_size * float(batch_size)
    scaled_bw_time = self.bw_time / self.profile_batch_size * \
        (batch_size if batch_size <= self.max_batch_size else self.max_batch_size)
    return scaled_fw_bw_time, scaled_bw_time

class ResNet50(object):
  def __init__(self):
   # model setting
   # time:ms size:MB
   self.max_batch_size = 128
   self.min_batch_size = 32
   self.weights = 100.0
   # stage 1 for min_batch_size
   self.cut_features = 0.4 * self.min_batch_size
   if FLAGS.only_pipe:
     self.comp_cut = [6.25] * 16
     self.devices_cut = [1] * 16
     self.weights_cut = [6.25] * 16
   else:
     stages = FLAGS.total_gpu_num / MAX_GPUS_PER_NODE
     scale = stages / 2
     self.comp_cut = [100.0 / scale, 100.0 / scale] * scale
     self.devices_cut = [1, 1] * scale
     self.weights_cut = [50 / scale, 50 / scale] * scale

  def compute_time(self, batch_size):
    # ms
    return 100.0 / 32 * float(batch_size)


class Toy(object):
  def __init__(self):
   # model setting
   # time:ms size:MB
   self.max_batch_size = 6
   self.min_batch_size = 2
   self.weights = 1280.0
   # stage 1 for min_batch_size
   self.cut_features = 1.5 * self.min_batch_size
   if FLAGS.only_pipe:
     self.comp_cut = [20.0] * 16
     self.weights_cut = [80.0] * 16
     self.devices_cut = [1] * 16
   else:
     self.comp_cut = [160.0] * 2
     self.weights_cut = [640.0] * 2
     self.devices_cut = [1] * 2

  def compute_time(self, batch_size):
    # ms
    return 160 * float(batch_size)

class AmoebaNet(object):
  def __init__(self):
   # model setting
   # time:ms size:MB
   self.max_batch_size = 32
   self.min_batch_size = 8
   self.weights = 700.0
   # cut size = 14*14*4096 *4B per element
   self.cut_features = 3.2 * self.min_batch_size
   self.four_stages = False
   self.fw_time = 60.0
   self.bw_time = 150.0
   self.fw_bw_time = self.fw_time + self.bw_time
   self.profile_batch_size = 8
   # The last layer's weight communication cannot be overlapped
   self.weights_of_last_layer = 33
   if not self.four_stages:
     #self.comp_cut = [106.0, 106.0]
     self.comp_cut = [120.0, 90.0]
     self.devices_cut = [1, 1]
     self.weights_cut = [160, 540]
   else:
     self.comp_cut = [53.0, 53.0] * 2
     self.devices_cut = [1, 1] * 2
     self.weights_cut = [80, 270] * 2

  def compute_time(self, batch_size):
    # ms
    scaled_fw_bw_time = self.fw_bw_time / self.profile_batch_size * float(batch_size)
    scaled_bw_time = self.bw_time / self.profile_batch_size * \
        (batch_size if batch_size <= self.max_batch_size else self.max_batch_size)
    return scaled_fw_bw_time, scaled_bw_time

class MegatronLM(object):
  def __init__(self):
   # model setting
   # time:ms size:MB
   self.max_batch_size = 8
   self.min_batch_size = 1 
   self.weights = 32000.0
   # stage 1 for min_batch_size
   self.cut_features = 4.5 * self.min_batch_size
   if FLAGS.DGX_2H:
     self.comp_cut = [58.0 * self.min_batch_size, 58.0 * self.min_batch_size] * 16
     self.devices_cut = [1, 1] * 16
     self.weights_cut = [1000, 1000] * 16
   else:
     self.comp_cut = [28.9 * self.min_batch_size, 28.9 * self.min_batch_size] * 32
     self.devices_cut = [1, 1] * 32
     self.weights_cut = [500, 500] * 32

  def compute_time(self, batch_size): # ms
    return 206 * 32 / 2 * float(batch_size)


class GNMT16(object):
  def __init__(self):
   # model setting
   # time:ms size:MB
   self.max_batch_size = 1024
   self.min_batch_size = 32 
   self.weights = 1179.0
   self.fw_time = 65.0
   self.bw_time = 110.0
   self.fw_bw_time = self.fw_time + self.bw_time
   self.profile_batch_size = 32.0
   # The last layer of GNMT16 is dense layer
   self.weights_of_last_layer = 150
   # stage 1 for min_batch_size
   self.cut_features = 3.0 / self.profile_batch_size * self.min_batch_size
   if FLAGS.only_pipe:
     assert FLAGS.total_gpu_num == 16
     self.comp_cut = [10.6] * FLAGS.total_gpu_num
     self.devices_cut = [1] * FLAGS.total_gpu_num
     self.weights_cut = [300] + [52] * (FLAGS.total_gpu_num - 2) + [150]
   else:
     self.comp_cut = [88.67, 81.33]
     self.devices_cut = [1, 1]
     self.weights_cut = [677.0, 502.0]

  def compute_time(self, batch_size):
    # ms
    scaled_fw_bw_time = self.fw_bw_time / self.profile_batch_size * float(batch_size)
    scaled_bw_time = self.bw_time / self.profile_batch_size * \
        (batch_size if batch_size <= self.max_batch_size else self.max_batch_size)
    return scaled_fw_bw_time, scaled_bw_time

if __name__ == '__main__':
  if FLAGS.model == "bert":
    model = BertLarge()
  elif FLAGS.model == "xlnet":
    model = XLNet()
  elif FLAGS.model == "vgg19":
    model = VGG19()
  elif FLAGS.model == "resnet50":
    model = ResNet50()
  elif FLAGS.model == "amoeba":
    model = AmoebaNet()
  elif FLAGS.model == "megatron":
    model = MegatronLM()
  elif FLAGS.model == "gnmt16":
    model = GNMT16()
  elif FLAGS.model == "toy":
    model = Toy()
  else:
    print("No model defined!")
    exit(1)
  # model setting
  # MB
  max_batch_size = model.max_batch_size
  min_batch_size = model.min_batch_size
  weights = model.weights
  # stage 1 for min_batch_size
  cut_features = model.cut_features
  comp_cut = model.comp_cut
  devices_cut = model.devices_cut
  weights_cut = model.weights_cut
  compute_time = model.compute_time
  #bw_time = model.bw_time
  # node setting
  # GB/s
  if FLAGS.server == "pipe-torch":
    # config (1)
    ethBdth_grpc = 1.25
    ethBdth_nccl = 1.25
  elif FLAGS.server == "p3.2xlarge":
    ethBdth_grpc = 1.25
    ethBdth_nccl = 1.25
  elif FLAGS.server == "p3.8xlarge":
    ethBdth_grpc = 0.32
    ethBdth_nccl = 1.2
  elif FLAGS.server == "p3.16xlarge":
    ethBdth_grpc = 0.8
    ethBdth_nccl = 3.0
  else:
    print("Unrecognized server type: %s" % FLAGS.server)
    exit(-1)

  if FLAGS.DGX_2H:
    ethBdth_grpc = 2.5
    ethBdth_nccl = 10.0
  pciBdth = 10.0
  nvBdth_all = 130.0
  nvBdth_half = 80.0
  nvBdth_p2p = 40.0

  # distribution setting
  GA = True

  # distribution strategy setting
  def dp(num_gpus_per_node, num_nodes, total_batch_size):
    ring_len = num_gpus_per_node * num_nodes
    batch_size = float(total_batch_size) / float(ring_len)
    print("DP: nodes: %d, gpus_per_node: %d" % (num_nodes, num_gpus_per_node))
    print("DP: total batch size: %d" % (batch_size * ring_len))
    print("DP: batch size: %d" % batch_size)
    if num_gpus_per_node == 1 and num_nodes == 1:
      print("DP: single gpu, no need to data parallel!")
      print("---------------------------------------------------")
      return -1.0
    if batch_size < min_batch_size:
      print("DP: too small batch size, or too many GPU cards")
      print("---------------------------------------------------")
      return -1.0

    if batch_size > max_batch_size:
      if not GA:
        print("DP: too large batch size, will be OOM")
        print("---------------------------------------------------")
        return -1.0
      else:
        GA_iters = (batch_size + max_batch_size - 1 ) / max_batch_size
    else:
      GA_iters = 1

    if FLAGS.server == "pipe-torch":
      # config (1) in paper
      bdth = 1.25
    elif FLAGS.server == "p3.2xlarge":
      bdth = 1.25 # up to 10Gbps
    elif num_nodes > 1:
      bdth = ethBdth_nccl
    elif num_gpus_per_node == MAX_GPUS_PER_NODE:
      bdth = nvBdth_all
    elif num_gpus_per_node == MAX_GPUS_PER_NODE / 2:
      bdth = nvBdth_half
    elif num_gpus_per_node == MAX_GPUS_PER_NODE / 4:
      bdth = nvBdth_p2p

    comp, bw_time = compute_time(batch_size)
    ar_vol = allreduce_vol(weights, ring_len)
    comm = ar_vol / bdth
    comm_never_overlapped = comm - bw_time
    if (comm_never_overlapped < 0):
      print("DP: WARNING: In the best case the allreduce comm time of params" +
          " can be perfectly overlapped by the bw compute time")
      comm_never_overlapped = allreduce_vol(model.weights_of_last_layer, ring_len) / bdth
    Q = comm / comp
    Q_best = comm_never_overlapped / comp
    eff = 1.0 / (1.0 + Q)
    if FLAGS.dp_best_overlap:
      eff = 1.0 / (1.0 + Q_best)
    print("DP: Ring Length: %d" % ring_len)
    print("DP: AllReduce Vol: %.4f" % ar_vol)
    print("DP: AllReduce bandwidth: %.4f GB/s" % bdth)
    print("DP: comm/comp (%.4f / %.4f) ratio Q: %.4f" % (comm, comp, Q))
    print("**DP: comm/comp with best bw computaion overlap (%.4f / %.4f)ratio Q: %.4f" \
        % (comm_never_overlapped, comp, Q_best))
    if GA:
      print("DP: GA iterations: %d" % GA_iters)
    print("DP: data parallel efficiency: %.4f" % eff)
    print("---------------------------------------------------")
    return eff, comp, comm

  def pipe(num_gpus_per_node, num_nodes, total_batch_size):
    pass

  def dapple(num_gpus_per_node, num_nodes, total_batch_size):
    nstages = len(comp_cut)
    ndev = np.sum(devices_cut)
    if num_gpus_per_node * num_nodes < ndev:
      print("Dapple: gpu not enough!")
      print("---------------------------------------------------")
      return -1.0
    max_ring_len = num_gpus_per_node * num_nodes / ndev
    ring_len = min(max(num_gpus_per_node  / max(devices_cut), 1), max_ring_len)
    unused = num_gpus_per_node * num_nodes - ring_len * ndev

    batch_size = float(total_batch_size) / float(ring_len)
    print("Dapple: nodes: %d, gpus_per_node: %d" % (num_nodes, num_gpus_per_node))
    print("Dapple: total batch size: %d" % (batch_size * ring_len))
    print("Dapple: batch size: %d" % batch_size)
    if num_gpus_per_node == 1 and num_nodes == 1:
      print("Dapple: single gpu, no need to data parallel!")
      print("---------------------------------------------------")
      return -1.0
    if batch_size < min_batch_size:
      print("Dapple: too small batch size, or too many GPU cards")
      print("---------------------------------------------------")
      return -1.0
    num_micro_batches = batch_size / min_batch_size
    print("Dapple: micro batch size: %d" % min_batch_size)
    print("Dapple: micro num batches per unit: %d" % num_micro_batches)
    print("Dapple: num of stages: %d" % nstages)
    if num_micro_batches < nstages:
      print("Dapple: too less micro batches to make pipeline full")
      print("---------------------------------------------------")
      return -1.0

    # placement
    if num_nodes <= 1:
      feat_bdth = nvBdth_p2p
    else:
      feat_bdth = ethBdth_grpc
    if num_nodes > nstages:
      ar_bdth  = ethBdth_nccl
    elif num_gpus_per_node == MAX_GPUS_PER_NODE:
      ar_bdth = nvBdth_all
    elif num_gpus_per_node == MAX_GPUS_PER_NODE / 2:
      ar_bdth = nvBdth_half
    elif num_gpus_per_node == MAX_GPUS_PER_NODE / 4:
      ar_bdth = nvBdth_p2p

    bubble = nstages - 1 + nstages - 1
    print("Dapple: bubble count with comm: %d" % bubble)
    fcomm = cut_features / feat_bdth
    print("Dapple: feat bandwidth: %0.2f GB/s" % feat_bdth)
    print("Dapple: feat comm: %.4f * 2.0" % fcomm)
    ### Take communication as one stage, and then get the slowest stage
    temp_cut = comp_cut + [2.0 * fcomm] * (nstages - 1)
    print("Dapple: temp cut %s " % temp_cut)
    max_comp_slice = np.argmax(temp_cut)
    if max_comp_slice >= nstages:
      print("feature map too large in pipeline!")
#      return -1.0
    comp_time = float(np.sum(temp_cut[:-(nstages-1)]))
    ## end2end execution time of pipeline with only only micro batch
    one_pipeline_time = float(np.sum(temp_cut))
    total_time = num_micro_batches * temp_cut[max_comp_slice] + \
                 one_pipeline_time - temp_cut[max_comp_slice]
    # activation communication time of total pipeline
    act_comm_time = fcomm * 2
    dpl_unit_Q = act_comm_time / (total_time - act_comm_time)
    print("Dapple: Pipeline Unit Q = comm/comp = %.4f" % dpl_unit_Q)
    print("Dapple: Pipeline Unit total time: %.4f ms" % total_time)
    dev_eff = []
    for i, dev in enumerate(devices_cut):
      for j in xrange(dev):
        deff = num_micro_batches * temp_cut[i] / total_time
        dev_eff.append(deff)
        if FLAGS.verbose:
          print("Dapple: dev %d efficiency: %.4f" % (np.sum(devices_cut[:i]) + j, deff))
    unit_eff = np.mean(dev_eff)
    print("Dapple: DP unit efficiency: %.4f" % unit_eff)
    if ring_len * np.max(devices_cut) == 1:
      eff = unit_eff
      print("Dapple: Just one Pipeline efficiency: %.4f" % eff)
      print("---------------------------------------------------")
    else:
      comp, bw_time = compute_time(batch_size) / unit_eff
      max_comm = 0.0
      for i in xrange(len(weights_cut)): 
        ar_vol = allreduce_vol(weights_cut[i], ring_len * devices_cut[i])
        if ring_len * devices_cut[i] > num_gpus_per_node:
          ar_bdth = ethBdth_grpc
        comm = ar_vol / ar_bdth
        max_comm = max(max_comm, comm)
      comm = max_comm
      # Now the total_time = all_micro_batch_fw_bw_time + apply_grad time
      total_time += comm
      Q = comm / comp
      dp_eff = 1.0 / (1.0 + Q)
      # overall comm/comp ratio of DAPPLE where pipeline and dp are mixtrued
      dpl_all_Q = (comm + act_comm_time) / comp
      print("Dapple: Ring Length: %d (%d unused)" % (ring_len, unused))
      print("Dapple: comm/comp (%.4f / %.4f) ratio Q: %.4f" % (comm, comp, Q))
      print("Dapple: **TOTAL comm/comp ratio Q: %.4f" % dpl_all_Q)
      print("Dapple: DP efficiency: %.4f" % dp_eff)
      eff = dp_eff * unit_eff
      print("Dapple: Dapple efficiency: %.4f" % eff)
      print("---------------------------------------------------")
    return eff, total_time

  tests_bak = [[8, 1], [4, 1], [2, 1], \
           [8, 2], [4, 2], [2, 2], \
           [8, 4], [4, 2], [2, 4]]
  tests = [[MAX_GPUS_PER_NODE, FLAGS.total_gpu_num / MAX_GPUS_PER_NODE]]
  #tests = tests_bak
  dp_dapple_out = []
  total_batch_size = FLAGS.total_batch_size
  if FLAGS.model == "amoeba":
    batches_bak = [128, 256, 512, 1024, 2048, 4096, 8192]
  elif FLAGS.model == "xlnet":
    batches_bak = [16, 32, 64, 128, 256, 512, 1024]
  elif FLAGS.model == "gnmt16":
    batches_bak = [512, 1024, 2048, 4096, 8192, 16384]
  elif FLAGS.model == "bert":
    batches_bak = [32, 64, 128, 256, 512, 1024, 2048]
  elif FLAGS.model == "resnet50":
    batches_bak = [512, 1024, 2048, 4096]
  else:
    print("FLAGS.model == %s" % FLAGS.model)
    batches_bak = None
  eff_per_batch = []
  for total_batch_size in batches_bak:
    for test in tests:
      ret_dp, comp_per_device, dp_comm = dp(test[0], test[1], total_batch_size)
      ret_dpl, dapple_total_time  = dapple(test[0], test[1], total_batch_size)
      print("DP Speed up over single device = %0.2f" % (comp_per_device*test[0]*test[1]/(comp_per_device + dp_comm)))
      print("Dapple Speed up over single device = %0.2f" % (comp_per_device*test[0]*test[1]/dapple_total_time))
      print("---------------------------------------------------")
      print("%d x %d\t%.2f\t%.2f" % (test[1], test[0], ret_dp, ret_dpl))
      dp_dapple_out.append((ret_dp, ret_dpl))
      if ret_dpl > ret_dp:
        print("###########################")
        print("dapple win!")
        print("###########################")
      else:
        print("###########################")
        print("data parallel win!")
        print("###########################")
    #print("configure\t dp_eff\t dapple_eff")
    #for i in xrange(len(dp_dapple_out)):
    #  #print("%d x %d\t%.2f\t%.2f" % (tests[i][0], tests[i][1], dp_dapple_out[i][0], dp_dapple_out[i][1]))
    #  print("%.2f\t%.2f" % (dp_dapple_out[i][0], dp_dapple_out[i][1]))

