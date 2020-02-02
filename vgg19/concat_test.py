import os

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
  devices = []
  num_replica = 8
  for i in range(num_replica):
    devices.append("/job:worker/replica:0/task:%d/device:GPU:%d" % i)
