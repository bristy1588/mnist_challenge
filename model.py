"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# The Step Function Multipliers MULT_A * MULT_B = 1

class Model(object):
  def __init__(self, MULT_A = 1.0, LAM_REG_WEIGHT = 0):

    print(' Step Size :', MULT_A, ' Regularization Weight: ', LAM_REG_WEIGHT)

    self.MULT_A = MULT_A
    self.MULT_B = 1.0 / MULT_A
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,32])
    b_conv1 = self._bias_variable([32])

    h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
    f_conv1 = tf.stop_gradient(tf.floor(h_conv1 * self.MULT_A) * self.MULT_B - h_conv1) + h_conv1
    reg_conv1 = tf.reduce_sum(tf.abs(h_conv1 - f_conv1 - self.MULT_B/2.0))

    h_pool1 = self._max_pool_2x2(f_conv1)

    # second convolutional layer
    W_conv2 = self._weight_variable([5,5,32,64])
    b_conv2 = self._bias_variable([64])

    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
    f_conv2 = tf.stop_gradient(tf.floor(h_conv2 * self.MULT_A) * self.MULT_B - h_conv2) + h_conv2
    reg_conv2 = tf.reduce_sum(tf.abs(h_conv2 - f_conv2 -  self.MULT_B/2.0))

    h_pool2 = self._max_pool_2x2(f_conv2)

    # first fully connected layer
    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    f_fc1 = tf.stop_gradient(tf.floor(h_fc1 * self.MULT_A) * self.MULT_B - h_fc1) + h_fc1
    reg_fc1 = tf.reduce_sum(tf.abs(h_fc1 - f_fc1 - self.MULT_B/2.0))

    # output layer
    W_fc2 = self._weight_variable([1024,10])
    b_fc2 = self._bias_variable([10])

    self.pre_softmax = tf.matmul(f_fc1, W_fc2) + b_fc2

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)

    self.reg_loss = reg_conv1 + reg_conv2 + reg_fc1
    self.my_loss = self.xent + self.reg_loss * LAM_REG_WEIGHT
    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
