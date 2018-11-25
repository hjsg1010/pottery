import os
import sys
import numpy as np
import h5py
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
# if not os.path.exists(DATA_DIR):
#  os.mkdir(DATA_DIR)
# if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#  www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#  zipfile = os.path.basename(www)
#  os.system('wget %s; unzip %s' % (www, zipfile))
#  os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#  os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
      Input:
        data: B,N,... numpy array
        label: B,... numpy array
      Return:
        shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
      rotation is per shape based along up direction
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -
                         angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_with_45s(data):
    """ Rotate the point clouds with the given angle
      Input:
        BxNx3 array, original of point clouds
      Return:
        BxNx3 array, rotated batch(8) of point clouds
    """
    #rotated_data = np.zeros(data.shape, dtype=np.float32)
    stacked_data = []
    stacked_data.append(data)
    for k in [45, 90, 135, 180, 225, 270, 315]:  # np.pi?
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(k), -np.sin(k)],
                       [0, np.sin(k), np.cos(k)]])
	# Ry does not help here. 
        """
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        """
        #R = np.dot(Rz, np.dot(Ry, Rx))
        rotated_data = np.dot(data, Rx)
        stacked_data.append(rotated_data)
    for k in [45, 90, 135, 180, 225, 270, 315]:
        Rz = np.array([[np.cos(k), -np.sin(k), 0],
                       [np.sin(k), np.cos(k), 0],
                       [0, 0, 1]])
        rotated_data = np.dot(data, Rz)
        stacked_data.append(rotated_data)

    #print(np.array(stacked_data).shape)
    return stacked_data #rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    # print(h5_filename)
    f = h5py.File('./data/'+h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    # seg = f['seg'][:]
    return (data, label)


def loadDataFile(filename):
    # print(filename)
    return load_h5(filename)


def load_h5_data_label_seg(h5_filename):
    # print(h5_filename)
    f = h5py.File('./data/'+h5_filename)
    data = f['data'][:]  # (2048, 2048, 3)
    label = f['label'][:]  # (2048, 1)
    #seg = f['seg'][:]  # (2048, 4)
    seg = f['seg'][:]  # (2048, 4)
    return (data, label, seg)

def loadsegDataFile(filename):
    # print(filename)
    return load_h5_data_label_seg(filename)

def data_generator(datafile_list, batchsize):
    MAX = 100
    data_q = []
    label_q = []
    seg_q = []
    _50p = 0
    label_count = np.zeros([5], dtype=np.int32)
    seg_count = np.zeros([3], dtype=np.int32)

    fn_idx = 0
    while True:  #for fn in datafile_list:
        #print(fn)
        #if fn == "1_5_844579.h5":
        #    continue
        """
        if len(top_q) > ENOUGH and len(mid_q) > ENOUGH and len(bot_q) > ENOUGH:
            #print(len(top_q), len(mid_q), len(bot_q))
            yield (top_q, mid_q, bot_q)
        """
        if len(data_q) >= batchsize:
            #random.shuffle(data_q)
            yield data_q[0:batchsize], label_q[0:batchsize], seg_q[0:batchsize]
            del data_q[0:batchsize]
            del label_q[0:batchsize]
            del seg_q[0:batchsize]
        #if len(data_q) > batchsize:
        #    #random.shuffle(data_q)
        #    data_batch = rotate_point_cloud_with_45s(data_q[0])
        #    print
        #    yield data_batch, label_q[0:1], seg_q[0:1]
        #    del data_q[0]
        #    del label_q[0]
        #    del seg_q[0]
        else:
            while len(data_q) < MAX and fn_idx < len(datafile_list):
                data, label, seg = loadsegDataFile(datafile_list[fn_idx])
                # (20, 2048, 3), (20,), (20, 4)
                seg[seg<=0] = 0.0000001
                seg[seg>=1] = 0.9999999
                #print(data.shape, label.shape, seg.shape)

                """
                for si in range(label.shape[0]):
                    if label[si] == 0 or label[si] == 4:  # pot #1 or #5
                        if seg[si][0] == 1 and len(bot_q) < MAX:  # bottom
                            bot_q.append([data[si], label[si], seg[si]])
                        elif seg[si][3] == 1 and len(top_q) < MAX:  # top
                            top_q.append([data[si], label[si], seg[si]])
                        elif seg[si][0] == 0 and seg[si][3] ==0 and len(mid_q) < MAX:  # mid
                            mid_q.append([data[si], label[si], seg[si]])
                    else:  # pot #2, #3, #4
                        if seg[si][0] == 1 and len(bot_q) < MAX:  # bottom
                            bot_q.append([data[si], label[si], seg[si]])
                        elif seg[si][2] == 1 and len(top_q) < MAX:  # top
                            top_q.append([data[si], label[si], seg[si]])
                        elif seg[si][0] == 0 and seg[si][2] ==0 and len(mid_q) < MAX:  # mid
                            mid_q.append([data[si], label[si], seg[si]])
                """
                file_size = int(datafile_list[fn_idx].split('_')[1])
                #print(fn, file_size)
                for i in range(file_size):
                    """
                    data_q.append(data[i])
                    label_q.append(label[i])
                    seg_q.append(seg[i])
                    """
                    if _50p < 2:
                        if seg[i][0] > 0.95 or seg[i][2] < 0.05:
                            data_q.append(data[i])
                            label_q.append(label[i])
                            label_count[label[i]] += 1
                            seg_q.append(seg[i])
                            if seg[i][0] > 0.95:
                                seg_count[0] += 1
                            if seg[i][2] < 0.05:
                                seg_count[2] += 1
                            _50p += 1
                    else:
                        data_q.append(data[i])
                        label_q.append(label[i])
                        label_count[label[i]] += 1
                        seg_q.append(seg[i])
                        seg_count[1] += 1
                        _50p = 0
                fn_idx += 1
            # shuffle with the same random seed
            rand_seed = random.randint(0,1000000)
            random.seed(rand_seed)
            random.shuffle(data_q)
            random.seed(rand_seed)
            random.shuffle(label_q)
            random.seed(rand_seed)
            random.shuffle(seg_q)
        if fn_idx >= len(datafile_list) and len(data_q) < batchsize:
            print("label_count:", label_count, "seg_count:", seg_count)
            yield None, None, None

#loadsegDataFile("3_20_48719.h5")
