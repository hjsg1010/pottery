from transform_nets import input_transform_net
import tf_util
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(
        tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


# def get_model(point_cloud, filters, is_training, bn_decay=None):
def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    k = 20
    #print(batch_size, num_point)

    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

#  print(edge_feature.shape)
    with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
        transform = input_transform_net(
            edge_feature, is_training, bn_decay, K=3)

    point_cloud_transformed = tf.matmul(point_cloud, transform)
    adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(
        point_cloud_transformed, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         # bn=True, is_training=is_training,
                         bn=False,
                         scope='dgcnn1', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net

    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         # bn=True, is_training=is_training,
                         bn=False,
                         scope='dgcnn2', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net

    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         # bn=True, is_training=is_training,
                         bn=False,
                         scope='dgcnn3', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net

    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         # bn=True, is_training=is_training,
                         bn=False,
                         scope='dgcnn4', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net4 = net

    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         # bn=True, is_training=is_training,
                         bn=False,
                         scope='agg', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=1, keep_dims=True)

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    # print(net)

    # 1 sum  shards' feature (except additional padding shards)
    # print(filters)
#  net = tf.multiply(net, filters)   # remove additional padding shards
    net = tf.reduce_sum(net, 0, keep_dims=True)



print("reduce_sum", net.shape)
print(net)


# 2 average shards' featre
#  net = tf.reduce_mean(net,0, kee_dims=True)

# 3 mutiply transpose matrix : B*1024 X 1024*B -> B*B or 1024*B X B*1024 -> 1024*1024
#  net = tf.matmul(net,net,transpose_b=True) #shape=B*B
#  net = tf_util.conv2d(net,1024,[1,1],padding='VALID', stride=[1,1], bn=True, is_training=is_training, scope='agg', bn_decay=bn_decay)p
#  print(net.shape)


#  print(net.shape)

net = skip_dense(net, 1024, 10, 0.1, is_training)
print("skip_dense: ", net.shape)


net = skip_dense(net, 1024, 10, 0.1, is_training)

#net = tf.layers.batch_normalization(net, training=is_training)
#net = tf.nn.relu(net)
net = tf.contrib.layers.fully_connected(
    net, 512, activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE, scope='fc1')
net = tf.contrib.layers.dropout(
    net, keep_prob=0.5, is_training=is_training, scope='dp1')
net = tf.contrib.layers.fully_connected(
    net, 256, activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE, scope='fc2')
net = tf.contrib.layers.dropout(
    net, keep_prob=0.5, is_training=is_training, scope='dp2')
net = tf.contrib.layers.fully_connected(
    net, 5, activation_fn=tf.nn.sigmoid, scope='fc3') # change activation function to sigmoid
print("final net: ", net.shape)
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
    # to reduce repeating same labels
    labels = tf.reduce_mean(labels, 0, keep_dims=True)
    #print(labels.get_shape(), pred.get_shape())

    #loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=pred))
    reg_loss = tf.reduce_mean(tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = classify_loss + reg_loss
    return loss


def skip_dense(x, size, repeat, scale, training):
    # in: [batch, size]
    for i in range(repeat):
        with tf.variable_scope("resnet_dense"+str(i)+"_"):
            #bn0 = tf.layers.batch_normalization(x, training=training)
            #act0 = tf.nn.relu(bn0)
            # dense1 = tf.layers.dense(act0, size,
            dense1 = tf.layers.dense(x, size,
                                     activation=None,
                                     # kernel_initializer=tf.random_uniform_initializer(minval=0.05),
                                     bias_initializer=tf.zeros_initializer,
                                     kernel_regularizer=tf.nn.l2_loss)
            dropout1 = tf.layers.dropout(dense1, rate=0.5, training=training)
            act1 = tf.nn.relu(dropout1)
        #x += dense1 * scale
        x += act1 * scale
        print("skip_dense"+str(i)+": ", x)
    return x


if __name__ == '__main__':
    batch_size = 2
    num_pt = 124
    pos_dim = 3

    input_feed = np.random.rand(batch_size, num_pt, pos_dim)
    label_feed = np.random.rand(batch_size)
    label_feed[label_feed >= 0.5] = 1
    label_feed[label_feed < 0.5] = 0
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
            print(res1.shape)
            print(res1)

            print(res2.shape)
            print(res2)
