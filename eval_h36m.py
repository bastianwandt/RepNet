"""
Implementation of the Paper from Wandt and Rosenhahn
"RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation"

This is the evaluation script for the Human3.6M dataset.
The original evaluation script used in the Paper was implemented in Matlab.
For simplicity the evaluation script was rewritten in Python, so expect minor differences in values due to rounding inaccuracies and a slightly improved network design.

For further information contact Bastian Wandt (wandt@tnt.uni-hannover.de)
"""

import os
import sys
import scipy.io as sio
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import load_model
import numpy as np
import numpy.matlib
import time
import glob
from eval_functions import err_3dpe
import h5py
from plot17j import plot17j


net_name = 'repnet_h36m_17j_late'

# path to Human3.6M 3D data
path_H36M = './data/Human36M/'

gen = load_model('models/generator_' + net_name + '.h5')

actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting',
           'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']

subjects = [9, 11]

# select a desired action and a subject from lists
action = actions[2]
sub = subjects[1]

files = glob.glob(path_H36M + 'S' + str(sub) + '/StackedHourglass/' + action + '*.h5')

# select only the first sequence for demonstration
file = files[0]
fname = os.path.basename(file)
fname = fname.replace("_", " ")

f = h5py.File(file, 'r')
data = list(f[u'poses'])

poses_2d = np.zeros((len(data), 32))

for p_idx in range(len(data)):
    p1 = data[p_idx]
    # reshape to my joint representation
    pose_2d = p1[[2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 12, 11, 10], :].T

    mean_x = np.mean(pose_2d[0, :])
    pose_2d[0, :] = pose_2d[0, :] - mean_x

    mean_y = np.mean(pose_2d[1, :])
    pose_2d[1, :] = pose_2d[1, :] - mean_y

    pose_2d = np.hstack((pose_2d[0, :], pose_2d[1, :]))
    poses_2d[p_idx, :] = pose_2d / np.std(pose_2d)

pred = gen.predict(poses_2d)

# visualize the prediction
plot17j(pred)

