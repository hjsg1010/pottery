import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from transform_nets_pointnet import input_transform_net, feature_transform_net


def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
#  pointclouds_pl = tf.placeholder(tf.float32, shape=(None, num_point, 3))
#  labels_pl = tf.placeholder(tf.int32, shape=(None))
  return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  end_points = {}

  with tf.variable_scope('transform_net1') as sc:
      transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
  point_cloud_transformed = tf.matmul(point_cloud, transform)
  input_image = tf.expand_dims(point_cloud_transformed, -1)

  net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

  with tf.variable_scope('transform_net2') as sc:
      transform = feature_transform_net(net, is_training, bn_decay, K=64)
  end_points['transform'] = transform
  net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
  net_transformed = tf.expand_dims(net_transformed, [2])

  net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
  net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

  net = tf.reshape(net, [batch_size, -1])
  
  #1 sum  shards' feature
  net = tf.reduce_sum(net,0, keep_dims=True)
  
  #2 average shards' featre
#  net = tf.reduce_mean(net,0, kee_dims=True)

  #3 mutiply transpose matrix : B*1024 X 1024*B -> B*B or 1024*B X B*1024 -> 1024*1024  
#  net = tf.matmul(net,net,transpose_b=True) #shape=B*B
#  net = tf_util.conv2d(net,1024,[1,1],padding='VALID', stride=[1,1], bn=True, is_training=is_training, scope='agg', bn_decay=bn_decay)
#  print(net.shape)
  
  
#  print(net.shape)
  
  net = tf.contrib.layers.fully_connected(net,512,activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE, scope='fc1')
  net = tf.contrib.layers.dropout(net,keep_prob=0.5, is_training=is_training, scope='dp1')
  net = tf.contrib.layers.fully_connected(net,256,activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE, scope='fc2')
  net = tf.contrib.layers.dropout(net,keep_prob=0.5, is_training=is_training, scope='dp2')
  net = tf.contrib.layers.fully_connected(net,5,activation_fn=None,scope='fc3')
#  print(net.shape)


#  net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
#                                scope='fc1', bn_decay=bn_decay) #shape = batch * 512
#  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
#                         scope='dp1') #shape = batch * 512                    
#  net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
#                                scope='fc2', bn_decay=bn_decay) #shape = batch * 256
#  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
#                        scope='dp2') #shape = batch * 256
#  net = tf_util.fully_connected(net, 5, activation_fn=None, scope='fc3') #shape = batch * 5

  return net, end_points

  
def get_loss(pred, label, end_points):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=5)
  labels = tf.reduce_mean(labels,0, keep_dims=True)
#  print(labels.shape)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  # # np.save('./debug/input_feed.npy', input_feed)
  # input_feed = np.load('./debug/input_feed.npy')
  # print input_feed

  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
      print (res1.shape)
      print (res1)

      print (res2.shape)
      print (res2)












