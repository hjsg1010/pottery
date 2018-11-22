import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='dgcnn_pottery_seg',
                    help='Model name: dgcnn_pottery, dgcnn+skipdense, dgcnn_pottery_seg')
parser.add_argument('--weight', default='', help='saved model weight file name.')
parser.add_argument('--log_dir', default='log_seg', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048,
                    help='Point Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=20,
                    help='Batch Size during training [default: 20]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum parameter [default: 0.9]')
parser.add_argument('--decay_step', type=int, default=200000,
                    help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8,
                    help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
WEIGHT = FLAGS.weight

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()


TRAIN_FILES = provider.getDataFiles('./data/train_files.txt')
TEST_FILES = provider.getDataFiles('./data/test_files.txt')

val_loss = None
val_loss_summary = tf.Summary()
val_loss_c = None
val_loss_c_summary = tf.Summary()
val_acc_c = None
val_acc_c_summary = tf.Summary()

"""
path_pot = "../lathe-and-shatter2/pot_class_PoC/"
raw_pottery_1=np.load(path_pot + 'one.npy')
raw_pottery_2=np.load(path_pot + 'two.npy')
raw_pottery_3=np.load(path_pot + 'three.npy')
raw_pottery_4=np.load(path_pot + 'four.npy')
raw_pottery_5=np.load(path_pot + 'five.npy')

pottery_1=np.reshape(raw_pottery_1,(1,2048,3))
pottery_2=np.reshape(raw_pottery_2,(1,2048,3))
pottery_3=np.reshape(raw_pottery_3,(1,2048,3))
pottery_4=np.reshape(raw_pottery_4,(1,2048,3))
pottery_5=np.reshape(raw_pottery_5,(1,2048,3))

pottery=np.concatenate((pottery_1,pottery_2,pottery_3,pottery_4,pottery_5),axis=0)
"""


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        # batch * BATCH_SIZE,  # Current index into the dataset.
                        batch,
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    # CLIP THE LEARNING RATE!
    learning_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch,  # batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
			# batchsize = 1
            pointclouds_pl, labels_pl, labels_pl_c = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            print(pointclouds_pl, labels_pl)
            # pointclouds_pl, labels_pl = MODEL.placeholder_inputs(None, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)

            # filters for real shards and padding shards
			
            #filters = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1024])

            # Get model and loss
            pred_c, pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            #pred, end_points = MODEL.get_model(pointclouds_pl, filters, is_training_pl, bn_decay=bn_decay)
            # print("pred.shape: ", pred.shape, "label.shape: ", labels_pl.shape)
            # print("label.shape: ", labels_pl.shape, "filter.shape: ", filters.shape)
            # print(filters)

            #loss = MODEL.get_loss(logits, labels_pl, end_points)
            loss, cls_loss, cls_loss_c, reg_loss = MODEL.get_loss(pred, pred_c, labels_pl, labels_pl_c, end_points)
            # loss = MODEL.get_seg_loss(pred, label, end_points)
            #tf.summary.scalar('loss', loss)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('cls_loss', cls_loss)
            tf.summary.scalar('cls_loss_c', cls_loss_c)
            tf.summary.scalar('reg_loss', reg_loss)

            #print("pred.shape: ", pred.shape, "label.shape: ", tf.to_int64(labels_pl).shape)
            # correct = tf.equal(tf.argmax(pred, 0), tf.to_int64(labels_pl))
            #correct = tf.equal(tf.to_int64(pred),tf.to_int64(labels_pl))
            #print("correct: ", correct)

            #accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))
            # tf.summary.scalar('accuracy', accuracy)
            #total_acc_summary.value.add(tag='train_accuracy', simple_value=total_acc)
            #val_acc_summary.value.add(tag='test_accuracy', simple_value=val_acc)
            val_loss_summary.value.add(tag='test_cls_loss', simple_value=val_loss)
            val_loss_c_summary.value.add(tag='test_cls_loss_c', simple_value=val_loss_c)
            val_acc_c_summary.value.add(tag='test_cls_acc_c', simple_value=val_acc_c)

#            filter_summary=tf.summary.image(filter)
            # Get training operator
            learning_rate = get_learning_rate(batch)
            # tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=0.1, momentum=0.9)
            elif OPTIMIZER == 'adam':
                # optimizer = tf.train.AdamOptimizer(learning_rate)
                optimizer = tf.train.AdamOptimizer()

            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=1.0)
            train_op = optimizer.minimize(loss, global_step=batch)

#            with tf.variable_scope('agg', reuse=True) as scope_conv:
#                w_conv=tf.get_variable('weights', shape=[1,1,320,1024])
#                weights=w_conv.eval()
#                tf.summary.histogram('histogram',weights)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_pl: True})

        # load model weight
        init_epoch = 0
        if WEIGHT == '':
            pass
        else:
            saver.restore(sess, WEIGHT)
            #init_epoch = int(WEIGHT.split("_epoch")[1].split("_acc")[0])
            init_epoch = 1
            print("Model restored:", WEIGHT)
            print("epoch:", init_epoch)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'labels_pl_c': labels_pl_c,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'pred_c': pred_c,
               'loss': loss,
               'cls_loss': cls_loss,
               'cls_loss_c': cls_loss_c,
               'reg_loss': reg_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               #'filters': filters,
              }

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_loss, eval_loss_c = eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            #if epoch % 10 == 0:
            if epoch % 1 == 0:  # save at every epoch
                save_path = saver.save(sess, os.path.join(
                    LOG_DIR, "model_"+str(epoch)+"_"+str(eval_loss)+"_"+str(eval_loss_c)+".ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files
    #train_file_idxs = np.arange(0, len(TRAIN_FILES))
    #np.random.shuffle(train_file_idxs)
    train_files = TRAIN_FILES
    #print(len(train_files))
    random.shuffle(train_files)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_labels = np.array([0,0,0], dtype=np.int32)

    gen = provider.data_generator(train_files, BATCH_SIZE)

    #for fn in range(len(TRAIN_FILES)):
    for fn in range(125000//BATCH_SIZE):  # total 125000 shards
        # log_string('----' + str(fn) + '-----')

        data_q, label_q, seg_q = next(gen)
        if data_q == None:
            print("an epoch is done in steps: ", fn)
            #print("total_labels:", total_labels)
            break
        #print(data_q[0].shape, label_q[0], seg_q[0].shape)
#        current_data, current_label, current_seg = provider.loadsegDataFile(train_files[fn])
#        current_data = current_data[:, 0:NUM_POINT, :]
#         current_data, current_label, current_seg_ = provider.shuffle_data(current_data, np.squeeze(current_label))
#        current_label = np.squeeze(current_label)
#        current_seg = np.squeeze(current_seg)

#        file_size = int(TRAIN_FILES[train_file_idxs[fn]].split('_')[1])
        file_size = BATCH_SIZE
        num_batches = 1

    #         # create filters
    #         ones = np.ones([file_size, 1024], np.int32)
    #         zeros = np.zeros([BATCH_SIZE-file_size, 1024], np.int32)
    # #        #print(ones.shape, zeros.shape)
    #         if file_size == 20:
    #             filters = ones
    #         else:
    #             filters = np.concatenate([ones, zeros], axis=0)
    #         # print(filters.shape)

            # # batch_idx => shard idx로 생각
            # # for batch_idx in range(num_batches):
            # batch_idx = 0
            # start_idx = batch_idx * BATCH_SIZE
            # end_idx = (batch_idx+1) * BATCH_SIZE

        current_data = np.array(data_q)
        current_label = np.array(label_q)
        current_seg = np.array(seg_q)
        #print(current_data.shape, current_label.shape, current_seg.shape)

        #rs_current_data=np.reshape(current_data,(20,2048,3))
        #rs_current_data=np.reshape(a_data_batch[i][0],(1,2048,3))
        #rs_seg = np.eye(2)[current_seg[i]] # rs_seg.shape=(4,2)
        #rs_seg0 = current_seg[i][0]*8+current_seg[i][1]*4+current_seg[i][2]*2+current_seg[i][3]
#            if a_data_batch[i][1] == 0 or a_data_batch[i][1] == 4:  # pot #1 or #4
#                rs_seg0 = a_data_batch[i][2][0]*4 + a_data_batch[i][2][3]
#                if a_data_batch[i][2][1] + a_data_batch[i][2][2] > 0:
#                    rs_seg0 = rs_seg0 + 2
#            else:  # pot #2 or #3 or #4
#                rs_seg0 = a_data_batch[i][2][0]*4 + a_data_batch[i][2][1]*2 + a_data_batch[i][2][2]

        
#            rs_seg = np.zeros([3,2], dtype=np.float32)
#            if current_label_i == 0 or current_label_i == 4:  # pot #1 or #4
            #rs_seg0 = current_seg[i][0]*4 + current_seg[i][3]
#                rs_seg[0][current_seg_i[0]] = 1
#                rs_seg[2][current_seg_i[2]] = 1
#                if current_seg_i[1] + current_seg_i[2] > 0:
                #rs_seg0 = rs_seg0 + 2
#                    rs_seg[1][1] = 1 
#                else:
#                    rs_seg[1][0] = 1
#            else:  # pot #2, #3, #4
            #rs_seg0 = current_seg[i][0]*4 + current_seg[i][1]*2 + current_seg[i][2]
#                rs_seg = np.eye(2)[current_seg_i[0:3]] # rs_seg.shape=(4,2)
        #print("rs_seg:", rs_seg.shape)

        #print(rs_seg)
        #rs_seg = np.eye(16)[rs_seg0] # rs_seg.shape=(4,2)
        #rs_seg = np.eye(8)[rs_seg0] # rs_seg.shape=(4,2)
        #print(rs_seg.shape, rs_seg)

        #if current_label_i == 0 or current_label_i == 4:  # pot #1 or #4
        #    if current_seg_i[0] == 1:
        #        rs_seg0 = 0
        #    elif current_seg_i[3] == 1:
        #        rs_seg0 = 2
        #    else:
        #        rs_seg0 = 1
        #else:
        #    if current_seg_i[0] == 1:
        #        rs_seg0 = 0
        #    elif current_seg_i[2] == 1:
        #        rs_seg0 = 2
        #    else:
        #        rs_seg0 = 1
        #rs_seg = np.eye(3)[rs_seg0]

        rs_label = np.eye(5)[current_label]
        #print(current_label.shape, rs_label.shape, current_label, rs_label)
        #sys.exit()
        rs_seg = current_seg

        #train_data=np.concatenate((rs_current_data, rs_pottery),axis=0)
        train_data=current_data

        # Augment batched point clouds by rotation and jittering
        jittered_data = provider.rotate_perturbation_point_cloud(train_data, angle_sigma= np.pi/2, angle_clip = np.pi)
        #rotated_data = provider.rotate_point_cloud(train_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        jittered_data = provider.random_scale_point_cloud(jittered_data)
        #jittered_data = provider.rotate_perturbation_point_cloud(jittered_data)
        jittered_data = provider.shift_point_cloud(jittered_data)
        #print(jittered_data.shape)
        #print(rs_seg.shape)

        feed_dict = {ops['pointclouds_pl']: jittered_data,
                    ops['labels_pl']: rs_seg,
                    ops['labels_pl_c']: rs_label,
                    ops['is_training_pl']: is_training
                    }
        
        summary, step, _, loss_val, pred_val, pred_c_val = sess.run([
            ops['merged'], 
            ops['step'],
            ops['train_op'],
            ops['loss'],
            #ops['cls_loss'],
            #ops['reg_loss'],
            ops['pred'],
            ops['pred_c'],
            ], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        
        #correct = np.sum(np.argmax(pred_val) == rs_seg0) 
        #print('pred, label, loss:', np.argmax(pred_val), rs_seg0, loss_val)
        #correct = np.prod(np.argmax(pred_val, axis=1) == np.argmax(rs_seg, axis=1))
        print('loss:', loss_val)
        #total_correct += correct
        total_seen += 1
        loss_sum += loss_val
        #total_labels[rs_seg0] += 1

        if fn % 100 == 100 - 1:
            # COMBINE FEATURES
            log_string('mean loss: %f' % (loss_sum / float(total_seen)))
            print('label:', rs_seg)
            print('pred:', pred_val)
            print('label_c:', current_label)
            print('pred_c:', np.argmax(pred_c_val, axis=1))
            print('loss_val:', loss_val)
            #log_string('accuracy: %f' % (total_correct / float(total_seen)))
        
            #total_acc_summary.value[0].simple_value=(total_correct/float(total_seen))    
            #train_writer.add_summary(total_acc_summary,step)

    #print("total_labels:", total_labels)


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    loss_sum_c = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_labels = np.array([0,0,0], dtype=np.int32)
    
    gen = provider.data_generator(TEST_FILES, BATCH_SIZE)

    #for fn in range(len(TEST_FILES)):
    for fn in range(125000//BATCH_SIZE):  # total 125000 shards
        
        # log_string('----' + str(fn) + '-----')
        #current_data, current_label, current_seg = provider.loadsegDataFile(TEST_FILES[fn])
        #current_data = current_data[:, 0:NUM_POINT, :]
        #current_label = np.squeeze(current_label)
        #current_seg = np.squeeze(current_seg)
        
        data_q, label_q, seg_q = next(gen)
        if data_q == None:
            print("an epoch is done in steps: ", fn)
            #print("total_labels:", total_labels)
            break

        #file_size = int(TEST_FILES[fn].split('_')[1])
        file_size = BATCH_SIZE
        num_batches = 1

        current_data = np.array(data_q)
        current_label = np.array(label_q)
        current_seg = np.array(seg_q)

        #rs_current_data=np.reshape(current_data,(1,2048,3))
        #rs_current_data=np.reshape(a_data_batch[i][0],(1,2048,3))
        #rs_pottery=np.reshape(pottery[current_label_i],(1,2048,3))
        #rs_pottery=np.reshape(pottery[a_data_batch[i][1]],(1,2048,3))
        #rs_seg = np.eye(2)[current_seg[i]] # rs_seg.shape=(4,2)
        #rs_seg0 = current_seg[i][0]*8+current_seg[i][1]*4+current_seg[i][2]*2+current_seg[i][3]
        #rs_seg0 = a_data_batch[i][2][0]*8+a_data_batch[i][2][1]*4+a_data_batch[i][2][2]*2+a_data_batch[i][2][3]
#            if a_data_batch[i][1] == 0 or a_data_batch[i][1] == 4:  # pot #1 or #4
#                rs_seg0 = a_data_batch[i][2][0]*4 + a_data_batch[i][2][3]
#                if a_data_batch[i][2][1] + a_data_batch[i][2][2] > 0:
#                    rs_seg0 = rs_seg0 + 2
#            else:  # pot #2 or #3 or #4
#                rs_seg0 = a_data_batch[i][2][0]*4 + a_data_batch[i][2][1]*2 + a_data_batch[i][2][2]

#            if current_label[i] == 0 or current_label[i] == 4:  # pot #1 or #4
#                rs_seg0 = current_seg[i][0]*4 + current_seg[i][3]
#                if current_seg[i][1] + current_seg[i][2] > 0:
#                    rs_seg0 = rs_seg0 + 2
#            else:  # pot #2, #3, #4
#                rs_seg0 = current_seg[i][0]*4 + current_seg[i][1]*2 + current_seg[i][2]

#            rs_seg = np.zeros([3,2], dtype=np.float32)
#            if current_label_i == 0 or current_label_i == 4:  # pot #1 or #4
            #rs_seg0 = current_seg[i][0]*4 + current_seg[i][3]
#                rs_seg[0][current_seg_i[0]] = 1
#                rs_seg[2][current_seg_i[2]] = 1
#                if current_seg_i[1] + current_seg_i[2] > 0:
                #rs_seg0 = rs_seg0 + 2
#                    rs_seg[1][1] = 1 
#                else:
#                    rs_seg[1][0] = 1
#            else:  # pot #2, #3, #4
            #rs_seg0 = current_seg[i][0]*4 + current_seg[i][1]*2 + current_seg[i][2]
#                rs_seg = np.eye(2)[current_seg_i[0:3]] # rs_seg.shape=(4,2)
        #print("rs_seg:", rs_seg)

        """
        if current_label_i == 0 or current_label_i == 4:  # pot #1 or #4
            if current_seg_i[0] == 1:
                rs_seg0 = 0
            elif current_seg_i[3] == 1:
                rs_seg0 = 2
            else:
                rs_seg0 = 1
        else:
            if current_seg_i[0] == 1:
                rs_seg0 = 0
            elif current_seg_i[2] == 1:
                rs_seg0 = 2
            else:
                rs_seg0 = 1

        rs_seg = np.eye(3)[rs_seg0]

        #print(rs_seg)
        #rs_seg = np.eye(16)[rs_seg0] # rs_seg.shape=(4,2)
        #rs_seg = np.eye(8)[rs_seg0] # rs_seg.shape=(4,2)
        #print(rs_seg.shape, rs_seg)
        """

        rs_label = np.eye(5)[current_label]
        rs_seg = current_seg

        #test_data=np.concatenate((rs_current_data, rs_pottery),axis=0)
        test_data=current_data

        #rotated_data = provider.rotate_point_cloud(test_data)
        jittered_data = provider.rotate_perturbation_point_cloud(test_data, angle_sigma=np.pi/2, angle_clip=np.pi)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        jittered_data = provider.random_scale_point_cloud(jittered_data)
        #jittered_data = provider.rotate_perturbation_point_cloud(jittered_data)
        jittered_data = provider.shift_point_cloud(jittered_data)

        feed_dict = {ops['pointclouds_pl']: jittered_data,
                    ops['labels_pl']: rs_seg,
                    ops['labels_pl_c']: rs_label,
                    ops['is_training_pl']: is_training
                    }
        #summary, 
        step, cls_loss, cls_loss_c, pred_val, pred_c_val = sess.run([
            #ops['merged'],
            ops['step'],
            ops['cls_loss'],
            ops['cls_loss_c'],
            ops['pred'],
            ops['pred_c'],
            ], feed_dict=feed_dict)

        #correct = np.sum(np.argmax(pred_val) == rs_seg0) 
        #print(np.argmax(pred_c_val, axis=1), np.argmax(rs_label, axis=1))
        correct = np.sum(np.argmax(pred_c_val, axis=1) == np.argmax(rs_label, axis=1))
        print('cls_loss, cls_loss_c, cls_acc_c:', cls_loss, cls_loss_c, correct)
        #print('pred, label, loss:', np.argmax(pred_val, axis=1), np.argmax(rs_seg, axis=1), loss_val)

        total_correct += correct / float(BATCH_SIZE)
        total_seen += 1
#            loss_sum += (loss_val*BATCH_SIZE)
        loss_sum += cls_loss
        loss_sum_c += cls_loss_c
        #total_labels[rs_seg0] += 1

        
    loss_avg = loss_sum / float(total_seen)
    loss_avg_c = loss_sum_c / float(total_seen)
    acc_avg_c = total_correct / float(total_seen)
    log_string('eval mean cls_loss: %f' % loss_avg)
    log_string('eval mean cls_loss_c: %f' % loss_avg_c)
    log_string('eval mean acc_loss_c: %f' % acc_avg_c)
    #log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
#    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    #val_acc_summary.value[0].simple_value=(total_correct/float(total_seen))    
    val_loss_summary.value[0].simple_value=(loss_avg)    
    val_loss_c_summary.value[0].simple_value=(loss_avg_c)    
    val_acc_c_summary.value[0].simple_value=(acc_avg_c)    
    test_writer.add_summary(val_loss_summary,step)
    test_writer.add_summary(val_loss_c_summary,step)
    test_writer.add_summary(val_acc_c_summary,step)
    #print("total_labels:", total_labels)
    print('label:', rs_seg)
    print('pred:', pred_val)
    print('cls_loss:', cls_loss)
    print('label_c:', current_label)
    print('pred_c:', np.argmax(pred_c_val, axis=1))
    print('cls_loss_c:', cls_loss_c)
    print('cls_acc_c:', correct)

    return loss_avg, loss_avg_c


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
