"""
Implementation of the Paper from Wandt and Rosenhahn
"RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation"

This training script trains a neural network similar to the paper.
Except some minor improvements that are documented in the code this is the original implementation.

For further information contact Bastian Wandt at wandt@tnt.uni-hannover.de
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import scipy.io as sio
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Model,load_model, Sequential
from keras.layers import Input, Dense, Activation, Lambda, Reshape, Flatten, concatenate, LeakyReLU
import numpy as np
import numpy.matlib
import keras.backend as K
import keras.layers as L
from keras.layers.merge import _Merge
from keras.optimizers import Adam
import random
from functools import partial
from eval_functions import err_3dpe


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def reprojection_layer(x):
    # reprojection layer as described in the paper

    x = tf.to_float(x)

    pose3 = tf.reshape(tf.slice(x, [0, 0], [-1, 48]), [-1, 3, 16])

    m = tf.reshape(tf.slice(x, [0, 48], [-1, 6]), [-1, 2, 3])

    pose2_rec = tf.reshape(tf.matmul(m, pose3), [-1, 32])

    return pose2_rec


def weighted_pose_2d_loss(y_true, y_pred):
    # the custom loss functions weights joints separately
    # it's possible to completely ignore joint detections by setting the respective entries to zero

    diff = tf.to_float(tf.abs(y_true - y_pred))

    # weighting the joints
    weights_t = tf.to_float(
        np.array([1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1]))

    weights = tf.tile(tf.reshape(weights_t, (1, 32)), (tf.shape(y_pred)[0], 1))

    tmp = tf.multiply(weights, diff)

    loss = tf.reduce_sum(tmp, axis=1) / 32

    return loss


def wasserstein_loss(y_true, y_pred):

    return K.mean(y_true * y_pred)


def cam_loss(y_true, y_pred):
    # loss function to enforce a weak perspective camera as described in the paper

    m = tf.reshape(y_pred, [-1, 2, 3])

    m_sq = tf.matmul(m, tf.transpose(m, perm=[0, 2, 1]))

    loss_mat = tf.reshape((2 / tf.trace(m_sq)), [-1, 1, 1])*m_sq - tf.eye(2)

    loss = tf.reduce_sum(tf.abs(loss_mat), axis=[1, 2])

    return loss


def kcs_layer(x):
    # implementation of the Kinematic Chain Space as described in the paper

    import tensorflow as tf

    # KCS matrix
    Ct = tf.constant([
          [1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
          [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 1, 0],
          [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0 , 0, 0, 0, 0,-1],
          [0, 0, 0, 0, -1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,-1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,-1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,-1, 0, 0]])

    C = tf.reshape(tf.tile(Ct, (tf.shape(x)[0], 1)), (-1, 16, 15))

    poses3 = tf.to_float(tf.reshape(x, [-1, 3, 16]))
    B = tf.matmul(poses3, C)
    Psi = tf.matmul(tf.transpose(B, perm=[0, 2, 1]), B)

    return Psi


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


net_name = 'repnet_h36m_17j'

print('training ' + net_name)

print('load training data...')
print('loading Human3.6M')
poses = sio.loadmat('data/tmp/Attributes_H36M_2d_3d_training_centralized_17j.mat')
poses_3d = poses['Att3d']/1000

# we directly train on 2D detections to learn the noise model of the detector
print('loading Stacked Hourglass detections')
poses_det = sio.loadmat('data/tmp/Attributes_H36M_2d_3d_training_sh_detections_17j.mat')
poses_det = poses_det['Att2d']
poses_2d = poses_det
poses_2d[:, 16:32] = -poses_2d[:, 16:32]

# randomly permute training data
rp = np.random.permutation(poses_3d.shape[0])
poses_3d = poses_3d[rp, :]

rp = np.random.permutation(poses_2d.shape[0])
poses_2d = poses_2d[rp, :]


# evaluate performance on a small subset of test data during training
print('load test data...')
poses_eval = sio.loadmat('data/tmp/Attributes_H36M_2d_3d_test_centralized_17j.mat')
poses_2d_eval = poses_eval['Att2d']
poses_2d_eval[:, 16:32] = -poses_2d_eval[:, 16:32]
poses_3d_eval = poses_eval['Att3d']/1000

print('done')

# setup training parameters
BATCH_SIZE = 32
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10
sz_set = poses_2d.shape[0]
num_joints = int(poses_2d.shape[1]/2)

# 2D -> 3D regression network
pose_in = Input(shape=(2*num_joints,))
l1 = Dense(1000)(pose_in)
l1 = LeakyReLU()(l1)

# in contrast to the paper we use this shared residual block for better performance
l21 = Dense(1000)(l1)
l21 = LeakyReLU()(l21)
l22 = Dense(1000)(l21)
l22 = L.add([l1, l22])
l22 = LeakyReLU()(l22)

# the following residual blocks are used just for 3D pose regression
l31 = Dense(1000)(l22)
l31 = LeakyReLU()(l31)
l32 = Dense(1000)(l31)
l32 = L.add([l22, l32])
l32 = LeakyReLU()(l32)

l41 = Dense(1000)(l32)
l41 = LeakyReLU()(l41)
l42 = Dense(1000)(l41)
l42 = L.add([l32, l42])
l42 = LeakyReLU()(l42)

l5 = Dense(1000)(l42)
l5 = LeakyReLU()(l5)

pose_out = Dense(3*num_joints)(l5)

# camera regression net
# in contrast to the paper we connect the camera regression network to the shared residual block for better performance
lc11 = Dense(1000)(l22)
lc11 = LeakyReLU()(lc11)
lc12 = Dense(1000)(lc11)
lc12 = L.add([l22, lc12])
lc12 = LeakyReLU()(lc12)

lc21 = Dense(1000)(lc12)
lc21 = LeakyReLU()(lc21)
lc22 = Dense(1000)(lc21)
lc22 = L.add([lc12, lc22])
lc22 = LeakyReLU()(lc22)

cam_out = Dense(6)(lc22)

# combine 3D pose and camera estimation
# it is later decomposed in the reprojection layer
concat_3d_cam = concatenate([pose_out, cam_out])

# connect the reprojection layer
rec_pose = Lambda(reprojection_layer)(concat_3d_cam)

# the critic network splits in two paths
# 1) a simple fully connected path
# 2) the path containing the KCS layer
d_in = Input(shape=(3*num_joints,))

# pose path
d1 = Dense(100)(d_in)
d1 = LeakyReLU()(d1)
d2 = Dense(100)(d1)
d2 = LeakyReLU()(d2)
d3 = Dense(100)(d2)
d3 = L.add([d1, d3])
d3 = LeakyReLU()(d3)
d6 = Dense(100)(d3)

# KCS path
psi = Lambda(kcs_layer)(d_in)
psi_vec = Flatten()(psi)
psi_vec = Dense(1000)(psi_vec)
psi_vec = LeakyReLU()(psi_vec)
d1_psi = Dense(1000)(psi_vec)
d1_psi = LeakyReLU()(d1_psi)
d2_psi = Dense(1000)(d1_psi)
d2_psi = L.add([psi_vec, d2_psi])

# we concatenate the two paths and add another FC layer
c_disc_vec = L.concatenate([d6, d2_psi])
d_last = Dense(100)(c_disc_vec)
d_last = LeakyReLU()(d_last)

d_out = Dense(1)(d_last)

# Now we initialize the two regression networks and the discriminator
cam_net = Model(inputs=pose_in, outputs=cam_out)
rep_net = Model(inputs=pose_in, outputs=rec_pose)
generator = Model(inputs=pose_in, outputs=pose_out)
discriminator = Model(inputs=d_in, outputs=d_out)

# from here we follow the Keras-team implementation of the improved Wasserstein GAN:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

# The generator_model is used when we want to train the generator layers.
# As such, we ensure that the discriminator layers are not trainable.
# Note that once we compile this model, updating .trainable will have no effect within it. As such, it
# won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
# as we compile the generator_model first.
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False

generator_input = Input(shape=(2*num_joints,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
rep_net_layers_for_generator = rep_net(generator_input)
cam_net_layers_for_generator = cam_net(generator_input)
adversarial_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator, rep_net_layers_for_generator, cam_net_layers_for_generator])
# We use the Adam paramaters from Gulrajani et al.
adversarial_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9), loss=[wasserstein_loss, weighted_pose_2d_loss, cam_loss], loss_weights=[1, 1, 1])

# Now that the generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

# The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
# The noise seed is run through the generator model to get generated images. Both real and generated images
# are then run through the discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
real_samples = Input(shape=poses_3d.shape[1:])
generator_input_for_discriminator = Input(shape=(2*num_joints,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

# We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
# We then run these samples through the discriminator as well. Note that we never really use the discriminator
# output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator(averaged_samples)

# The gradient penalty loss function requires the input averaged samples to get gradients. However,
# Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
# of the function with the averaged samples here.
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

# Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
# real samples and generated samples before passing them to the discriminator: If we had, it would create an
# output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
# would have only BATCH_SIZE samples.

# If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
# samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])
# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
# samples, and the gradient penalty loss for the averaged samples.
discriminator_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])
# We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
# negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
# gradient_penalty loss function and is not used.
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

# training starts here
# we mostly use the notation from:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
for epoch in range(20):
    np.random.shuffle(poses_3d)
    print("Epoch: ", epoch)
    print("Number of batches: ", int(poses_3d.shape[0] // BATCH_SIZE))
    discriminator_loss = []
    adversarial_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    for i in range(int(poses_2d.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        noise_minibatches = poses_2d[i * minibatches_size:(i + 1) * minibatches_size]
        # randomly sample from 3d poses
        rand_samples = random.sample(range(0, poses_3d.shape[0]), minibatches_size)
        discriminator_minibatches = poses_3d[rand_samples,:]
        for j in range(TRAINING_RATIO):
            pose_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            noise = noise_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]

            discriminator_loss.append(discriminator_model.train_on_batch([pose_batch, noise], [positive_y, negative_y, dummy_y]))

            adversarial_loss.append(adversarial_model.train_on_batch(noise, [np.matlib.ones((BATCH_SIZE, 1)), noise, np.matlib.zeros((BATCH_SIZE, 1))]))

        # visualize training progress for a small subset of test samples
        if i % 100 == 0 and i > 0:
            pred = generator.predict(poses_2d_eval[0:200, :])

            # calculate 3d pose estimation error 3DPE
            val = 0
            for p in range(200):
                val = val + 1000*err_3dpe(poses_3d_eval[p:p+1, :], pred[p:p+1, :])

            val = val/200

            sys.stdout.write("\rIteration %d: 3d error: %.3e, rep_err: %.3e, cam_err: %.3e, disc_loss: %.3e, gen_disc_loss: %.3e "
                             % (i, val, adversarial_loss[-1][2], adversarial_loss[-1][3], discriminator_loss[-1][0], adversarial_loss[-1][1]))
            try:
                with open("logs/log_" + net_name + ".txt", "a") as logfile:
                    logfile.write("%d \t %.3e \t %.3e \t %.3e \t %.3e \t %.3e \n"
                                 % (i, val, adversarial_loss[-1][2], adversarial_loss[-1][3], discriminator_loss[-1][0], adversarial_loss[-1][1]))
            except:
                print('error while writing logfile')

            sys.stdout.flush()

        # save model every 1000 iterations
        if i % 1000 == 0 and i > 0:
            generator.save('models/tmp/generator_' + net_name + '.h5')
            discriminator.save('models/tmp/discriminator_' + net_name + '.h5')
            adversarial_model.save('models/tmp/adversarial_model_' + net_name + '.h5')
            cam_net.save('models/tmp/cam_net_' + net_name + '.h5')

    # decrease learning rate every 5 epochs
    if epoch % 5 == 0 and epoch > 0:
        # set new learning rate for discriminator
        lrd = K.get_value(discriminator_model.optimizer.lr)
        lrd = lrd / 10
        K.set_value(discriminator_model.optimizer.lr, lrd)

        # set new learning rate for adversarial model
        lra = K.get_value(adversarial_model.optimizer.lr)
        lra = lra / 10
        K.set_value(adversarial_model.optimizer.lr, lra)


session.close()


