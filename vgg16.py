import tensorflow as tf
import numpy as np
"""
reference:
https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-16-transfer-learning/
https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/407_transfer_learning.py
"""


class Vgg16Feature:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def feature_loss(self, ground_true, inpainting):
        gt_conv1_1, gt_conv2_1, gt_conv3_1 = self._get_feature(ground_true)
        inp_conv1_1, inp_conv2_1, inp_conv3_1 = self._get_feature(inpainting)

        loss = self._frobenius_norm(gt_conv1_1, inp_conv1_1) + self._frobenius_norm(gt_conv2_1, inp_conv2_1) +\
               self._frobenius_norm(gt_conv3_1, inp_conv3_1)
        return loss

    def _frobenius_norm(self, x, y):
        loss = tf.square(x-y)
        loss = tf.sqrt(1e-5 + tf.reduce_sum(loss, axis=[1, 2, 3]))
        loss = tf.reduce_mean(loss)
        return loss

    def _get_feature(self, x):
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self._conv_layer(bgr, "conv1_1")
        conv1_2 = self._conv_layer(conv1_1, "conv1_2")
        pool1 = self._max_pool(conv1_2, 'pool1')

        conv2_1 = self._conv_layer(pool1, "conv2_1")
        conv2_2 = self._conv_layer(conv2_1, "conv2_2")
        pool2 = self._max_pool(conv2_2, 'pool2')

        conv3_1 = self._conv_layer(pool2, "conv3_1")

        return conv1_1, conv2_1, conv3_1