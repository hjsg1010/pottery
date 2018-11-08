# -*- coding: utf-8 -*-

import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='dgcnn_pottery_alt', help='Model name: pointnet')
parser.add_argument('--log_dir', default='log_res', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.8]')
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

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
#TRAIN_FILES = provider.getDataFiles( \
#    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
#TEST_FILES = provider.getDataFiles(\
#    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

#TRAIN_FILES=provider.getDataFiles('./data/outputs/h5/train_files.txt')
#TEST_FILES=provider.getDataFiles('./data/outputs/h5/test_files.txt')

TRAIN_FILES=provider.getDataFiles('./data/train_files.txt')
TEST_FILES=provider.getDataFiles('./data/test_files.txt')

val_acc=None
val_acc_summary=tf.Summary()
total_acc=None
total_acc_summary=tf.Summary()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch, #batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch, #batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


	
	
def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
			# batchsize = 1
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
#            tf.summary.scalar('bn_decay', bn_decay)
            
            # filters for real shards and padding shards
            filters = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE,1024]) #1024 = global filter length (./models/dgcnn~.py)

            # Get model and loss 
#            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            pred, end_points = MODEL.get_model(pointclouds_pl, filters, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)
#            val_acc_summary.value.add(tag='train_accuracy', simple_value=val_acc)
#            total_acc_summary.value.add(tag='total_accuracy',simple_value=total_acc)

#            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
#            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))
#            tf.summary.scalar('accuracy', accuracy)
            total_acc_summary.value.add(tag='train_accuracy', simple_value=total_acc)
            val_acc_summary.value.add(tag='test_accuracy', simple_value=val_acc)
            
            
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
#                optimizer = tf.train.AdamOptimizer(learning_rate)
                optimizer = tf.train.AdamOptimizer()
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
#        config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.allow_soft_placement = True
        config.log_device_placement = False
             
        
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'filters': filters,
               }

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_"+str(epoch)+".ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    total_correct=0
    total_seen=0
    loss_sum=0
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
#        print("current_label: ", current_label)
        
#        file_size = current_data.shape[0]
#        num_batches = file_size // BATCH_SIZE
        
        file_size=int(TRAIN_FILES[train_file_idxs[fn]].split('_')[1])
        num_batches=1
        
        #create filters
        ones = np.ones([file_size, 1024], np.int32)
        zeros = np.zeros([BATCH_SIZE - file_size, 1024], np.int32)
#        print(ones.shape, zeros.shape)
        if file_size == 20:
            filters = ones
        else:
            filters = np.concatenate([ones,zeros], axis=0)
#        print("filters.shape: ", filters.shape)

        # batch_idx => shard idx로 생각
#        for batch_idx in range(num_batches):
#            start_idx = batch_idx * BATCH_SIZE
#            end_idx = (batch_idx+1) * BATCH_SIZE
#            
#            # Augment batched point clouds by rotation and jittering
#            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
#            jittered_data = provider.jitter_point_cloud(rotated_data)
#            jittered_data = provider.random_scale_point_cloud(jittered_data)
#            jittered_data = provider.rotate_perturbation_point_cloud(jittered_data)
#            jittered_data = provider.shift_point_cloud(jittered_data)
#
#            feed_dict = {ops['pointclouds_pl']: jittered_data,
#                         ops['labels_pl']: current_label[start_idx:end_idx],
#                         ops['is_training_pl']: is_training,}
#            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
#                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
#            train_writer.add_summary(summary, step)
#            pred_val = np.argmax(pred_val, 1)
#            correct = np.sum(pred_val == current_label[start_idx:end_idx])
#            total_correct += correct
#            total_seen += BATCH_SIZE
#            loss_sum += loss_val
#        
#        
#        
#        
#        val_acc_summary.value[0].simple_value=(total_correct/float(total_seen))
#
#
#        train_writer.add_summary(val_acc_summary,step)
#        
#        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
#        log_string('accuracy: %f' % (total_correct / float(total_seen)))
#
#    total_acc_summary.value[0].simple_value=(total_total_correct/float(total_total_seen))    
#    train_writer.add_summary(total_acc_summary,step)
    
        batch_idx = 0
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        # Augment batched point clouds by rotation and jittering
        rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
        jittered_data = provider.jitter_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(jittered_data)
        jittered_data = provider.rotate_perturbation_point_cloud(jittered_data)
        jittered_data = provider.shift_point_cloud(jittered_data)

        feed_dict = {ops['pointclouds_pl']: jittered_data,
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,
                     ops['filters']: filters,
                    }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        
        
        #print('pred_val:', pred_val)
        pred_val = np.argmax(pred_val, 1)
        #print('pred, label, loss:', pred_val[0], current_label[0], loss_val)
        correct = np.sum(pred_val[0] == current_label[0])
        total_correct += correct
        total_seen += num_batches
        loss_sum += loss_val

        if fn % 100 == 100 - 1:
       	    # COMBINE FEATURES
            log_string('mean loss: %f' % (loss_sum / float(total_seen)))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
        
            total_acc_summary.value[0].simple_value=(total_correct/float(total_seen))    
            train_writer.add_summary(total_acc_summary,step)


#def eval_one_epoch(sess, ops, test_writer):
#    """ ops: dict mapping from string to tf ops """
#    is_training = False
#    total_correct = 0
#    total_seen = 0
#    loss_sum = 0
#    total_seen_class = [0 for _ in range(NUM_CLASSES)]
#    total_correct_class = [0 for _ in range(NUM_CLASSES)]
#    
#    for fn in range(len(TEST_FILES)):
#        log_string('----' + str(fn) + '-----')
#        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
#        current_data = current_data[:,0:NUM_POINT,:]
#        current_label = np.squeeze(current_label) - 1
##        print("current_label.shape: ", current_label.shape)
#    
#        file_size = current_data.shape[0]
#        num_batches = file_size // BATCH_SIZE
##        print("num_batches:",num_batches," file_size:",file_size," BATCH_SIZE:",BATCH_SIZE)
#        
#        for batch_idx in range(num_batches):
#            start_idx = batch_idx * BATCH_SIZE
#            end_idx = (batch_idx+1) * BATCH_SIZE
#
#
#            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
#                         ops['labels_pl']: current_label[start_idx:end_idx],
#                         ops['is_training_pl']: is_training}
#            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
#                ops['loss'], ops['pred']], feed_dict=feed_dict)
#            pred_val = np.argmax(pred_val, 1)
##            print("pred_val.shape: ", pred_val.shape)
##            print("pred_val.value: ", pred_val)
##            print("current_label index value: ", current_label[0])
##            current_label=tf.reduce_mean(current_label[start_idx:end_idx],0)
##            print("NEW current_label index shape: ", current_label.shape)
##            print("NEW current_label index value: ", current_label)
#            correct = np.sum(pred_val == current_label[start_idx:end_idx])
##            correct = np.sum(pred_val == current_label[start_idx])
#
#            total_correct += correct
#            total_seen += BATCH_SIZE
##            loss_sum += (loss_val*BATCH_SIZE)
#            loss_sum = loss_val
##            for i in range(start_idx, end_idx):
##                l = current_label[i]
##                total_seen_class[l] += 1
##                total_correct_class[l] += (pred_val[i-start_idx] == l)
#            
#    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
#    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
##    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for fn in range(len(TEST_FILES)):
        #log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
    
#        file_size = current_data.shape[0]
#        num_batches = file_size // BATCH_SIZE
#        print("num_batches:",num_batches," file_size:",file_size," BATCH_SIZE:",BATCH_SIZE)
        file_size = int(TEST_FILES[fn].split('_')[1])
        #print(TEST_FILES[fn], file_size)
        num_batches = 1

        # create filters
        ones = np.ones([file_size, 1024], np.int32)
        zeros = np.zeros([BATCH_SIZE-file_size, 1024], np.int32)
        #print(ones.shape, zeros.shape)
        if file_size == 20:
          filters = ones
        else:
          filters = np.concatenate([ones, zeros], axis=0)
        #print(filters.shape)
	       
        #for batch_idx in range(num_batches):
        batch_idx = 0
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE


        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,
                     ops['filters']: filters,
                    }
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
#            print("pred_val.shape: ", pred_val.shape)
#            print("pred_val.value: ", pred_val)
#            print("current_label index value: ", current_label[0])
        #print('pred_val, label:', pred_val[0], current_label[0])
#            current_label=tf.reduce_mean(current_label[start_idx:end_idx],0)
#            print("NEW current_label index shape: ", current_label.shape)
#            print("NEW current_label index value: ", current_label)
        #correct = np.sum(pred_val == current_label[start_idx:end_idx])
        correct = np.sum(pred_val[0] == current_label[0])
#            correct = np.sum(pred_val == current_label[start_idx])
        print('pred, label, loss:', pred_val[0], current_label[0], loss_val)

        total_correct += correct
        total_seen += num_batches
#            loss_sum += (loss_val*BATCH_SIZE)
        loss_sum += loss_val
#            for i in range(start_idx, end_idx):
#                l = current_label[i]
#                total_seen_class[l] += 1
#                total_correct_class[l] += (pred_val[i-start_idx] == l)
        
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
#    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    val_acc_summary.value[0].simple_value=(total_correct/float(total_seen))    
    test_writer.add_summary(val_acc_summary,step)
    
if __name__ == "__main__":
    train()
    LOG_FOUT.close()
