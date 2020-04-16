# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import mobilenet_v1
from nets import nasnet
from nets import inception_v4

slim = tf.contrib.slim

# def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
#                            window_size_ms, window_stride_ms,
#                            dct_coefficient_count):
#   """Calculates common settings needed for al7778888l models.
#
#   Args:
#     label_count: How many classes are to be recognized.
#     sample_rate: Number of audio samples per second.
#     clip_duration_ms: Length of each audio clip to be analyzed.
#     window_size_ms: Duration of frequency analysis window.
#     window_stride_ms: How far to move in time between frequency windows.
#     dct_coefficient_count: Number of frequency bins to use for analysis.
#
#   Returns:
#     Dictionary containing common settings.
#   """
#   desired_samples = int(sample_rate * clip_duration_ms / 1000)
#   window_size_samples = int(sample_rate * window_size_ms / 1000)
#   window_stride_samples = int(sample_rate * window_stride_ms / 1000)
#   length_minus_window = (desired_samples - window_size_samples)
#   if length_minus_window < 0:
#     spectrogram_length = 0
#   else:
#     spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
#   fingerprint_size = dct_coefficient_count * spectrogram_length
#   return {
#       'desired_samples': desired_samples,
#       'window_size_samples': window_size_samples,
#       'window_stride_samples': window_stride_samples,
#       'spectrogram_length': spectrogram_length,
#       'dct_coefficient_count': dct_coefficient_count,
#       'fingerprint_size': fingerprint_size,
#       'label_count': label_count,
#       'sample_rate': sample_rate,
#   }


def create_model(input, label_count, model_architecture,
                 is_training, runtime_settings=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  image_size specified in model_settings['image_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'conv':
    return create_conv_model(input, label_count, is_training)
  elif model_architecture == 'mobile':
    return create_mobilenet_model(input, label_count, is_training)
  elif model_architecture == 'mnist_mobile':
    return create_mnist_mobilenet_model(input, label_count, is_training)
  elif model_architecture == 'nasnet':
    return create_nasnet_model(input, label_count, is_training)
  elif model_architecture == 'inception':
    return create_inception_model(input, label_count, is_training)
  elif model_architecture == 'squeeze':
    return create_low_latency_squeeze_model(input, label_count, is_training)
  elif model_architecture == 'squeeze2':
    return create_low_latency_squeeze_model2(input, label_count, is_training)


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_conv_model(input, label_count, is_training):
  """Builds a mobilenet model.
  Here's the layout of the graph:

  (input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  Args:
    input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  fingerprint_4d = input

  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  #first_conv = BatchNorm(first_conv, is_training, name='bn1')
  first_relu = tf.nn.relu(first_conv)
  #first_relu = LeakyReLU(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias

  #second_conv = BatchNorm(second_conv, is_training, name='bn2')
  second_relu = tf.nn.relu(second_conv)
  #second_relu = LeakyReLU(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2] # 20
  second_conv_output_height = second_conv_shape[1] # 33

  # second_conv_element_count = 42240
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)

  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])

  # label_count = 12 = x + 2
  # label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_mobilenet_model(input, label_count, is_training):
  dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  logits, end_points = mobilenet_v1.mobilenet_v1(input, num_classes=label_count)

  return logits, dropout_prob


def create_nasnet_model(input, label_count, is_training):
  dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
    logits, end_points = nasnet.build_nasnet_mobile(input, label_count)

  return logits, dropout_prob


def create_inception_model(input, label_count, is_training):
  dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  logits, end_points = inception_v4.inception_v4(input, label_count)

  return logits, dropout_prob


def create_mnist_mobilenet_model(input, label_count, is_training):
  dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  X_img = tf.reshape(input, [-1, 28, 28, 1])
  resizing_img = tf.image.resize_bilinear(X_img, [224, 224])

  logits, end_points = mobilenet_v1.mobilenet_v1(resizing_img, label_count)
  return logits, dropout_prob


def create_low_latency_squeeze_model(input, label_count, is_training):
    squeeze_ratio = 1
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    # input_frequency_size = model_settings['dct_coefficient_count']
    # input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = input
    print('fingerprint_4d : ', fingerprint_4d)

    first_filter_width = 7
    first_filter_height = 7
    first_filter_count = 64
    first_weights = tf.get_variable("first_weight", shape=[first_filter_height, first_filter_width, 1, first_filter_count],
                                    initializer=tf.contrib.layers.xavier_initializer())

    # conv1_1
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 2, 2, 1], 'SAME')
    print('first_conv : ', first_conv)
    relu1 = tf.nn.relu(first_conv + bias_variable([64]))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print('pool1 : ', pool1)
    fire2 = fire_module('fire2', pool1, squeeze_ratio * 16, 64, 64)
    print('fire2 : ', fire2)
    fire3 = fire_module('fire3', fire2, squeeze_ratio * 16, 64, 64, True)
    print('fire3 : ', fire3)
    pool3 = tf.nn.max_pool(fire3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print('pool3 : ', pool3)
    fire4 = fire_module('fire4', pool3, squeeze_ratio * 32, 128, 128)
    print('fire4 : ', fire4)
    fire5 = fire_module('fire5', fire4, squeeze_ratio * 32, 128, 128, True)
    print('fire5 : ', fire5)
    pool5 = tf.nn.max_pool(fire5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    fire6 = fire_module('fire6', pool5, squeeze_ratio * 48, 192, 192)

    fire7 = fire_module('fire7', fire6, squeeze_ratio * 48, 192, 192, True)

    fire8 = fire_module('fire8', fire7, squeeze_ratio * 64, 256, 256)

    fire9 = fire_module('fire9', fire8, squeeze_ratio * 64, 256, 256, True)

    # 50% dropout
    dropout9 = tf.nn.dropout(fire9, dropout_prob)
    print('dropout9 : ', dropout9)
    second_weights = tf.Variable(tf.random_normal([1, 1, 512, 1000], stddev=0.01), name="second_weight")
    second_conv = tf.nn.conv2d(dropout9, second_weights, [1, 1, 1, 1], 'SAME')
    print('second_conv : ', second_conv)
    relu10 = tf.nn.relu(second_conv + bias_variable([1000]))
    print('relu10 : ', relu10)
    # avg pool
    pool10 = tf.nn.avg_pool(relu10, ksize=[1, 13, 13, 1], strides=[1, 1, 1, 1], padding='VALID')
    print('pool10 : ', pool10)
    last_conv_shape = pool10.get_shape()
    last_conv_ouput_width = last_conv_shape[2]
    last_conv_ouput_height = last_conv_shape[1]
    last_conv_element_count = int(last_conv_ouput_width * last_conv_ouput_height * 1000)
    flattend_last_conv = tf.reshape(pool10, [-1, last_conv_element_count])
    print('last_conv_element_count', last_conv_element_count)
    print('flattend_last_conv', flattend_last_conv)
    print('label_count', label_count)
    final_fc_weights = tf.get_variable("final_fc_weights", shape=[last_conv_element_count, label_count], initializer=tf.contrib.layers.xavier_initializer())
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattend_last_conv, final_fc_weights) + final_fc_bias

    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_low_latency_squeeze_model2(input, label_count, is_training):
    squeeze_ratio = 1
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    # input_frequency_size = model_settings['dct_coefficient_count']
    # input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = input
    print('fingerprint_4d : ', fingerprint_4d)

    first_filter_width = 7
    first_filter_height = 7
    first_filter_count = 64
    first_weights = tf.get_variable("first_weight", shape=[first_filter_height, first_filter_width, 1, first_filter_count],
                                    initializer=tf.contrib.layers.xavier_initializer())

    # conv1_1
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 2, 2, 1], 'SAME')
    print('first_conv : ', first_conv)
    relu1 = tf.nn.relu(first_conv + bias_variable([64]))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print('pool1 : ', pool1)
    fire2 = fire_module('fire2', pool1, squeeze_ratio * 16, 64, 64)
    print('fire2 : ', fire2)
    fire3 = fire_module('fire3', fire2, squeeze_ratio * 16, 64, 64, True)
    print('fire3 : ', fire3)
    fire4 = fire_module('fire4', fire3, squeeze_ratio * 32, 128, 128)
    print('fire4 : ', fire4)

    pool4 = tf.nn.max_pool(fire4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print('pool4 : ', pool4)

    fire5 = fire_module('fire5', pool4, squeeze_ratio * 32, 128, 128, True)
    print('fire5 : ', fire5)
    fire6 = fire_module('fire6', fire5, squeeze_ratio * 48, 192, 192)

    fire7 = fire_module('fire7', fire6, squeeze_ratio * 48, 192, 192, True)

    fire8 = fire_module('fire8', fire7, squeeze_ratio * 64, 256, 256)

    pool5 = tf.nn.max_pool(fire8, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    fire9 = fire_module('fire9', pool5, squeeze_ratio * 64, 256, 256, True)

    # 50% dropout
    dropout9 = tf.nn.dropout(fire9, dropout_prob)

    print('dropout9 : ', dropout9)
    second_weights = tf.Variable(tf.random_normal([1, 1, 512, 1000], stddev=0.01), name="second_weight")
    second_conv = tf.nn.conv2d(dropout9, second_weights, [1, 1, 1, 1], 'SAME')
    print('second_conv : ', second_conv)
    relu10 = tf.nn.relu(second_conv + bias_variable([1000]))
    print('relu10 : ', relu10)
    # avg pool
    pool10 = tf.nn.avg_pool(relu10, ksize=[1, 13, 13, 1], strides=[1, 1, 1, 1], padding='VALID')
    print('pool10 : ', pool10)
    last_conv_shape = pool10.get_shape()
    last_conv_ouput_width = last_conv_shape[2]
    last_conv_ouput_height = last_conv_shape[1]
    last_conv_element_count = int(last_conv_ouput_width * last_conv_ouput_height * 1000)
    flattend_last_conv = tf.reshape(pool10, [-1, last_conv_element_count])
    print('last_conv_element_count', last_conv_element_count)
    print('flattend_last_conv', flattend_last_conv)
    print('label_count', label_count)
    final_fc_weights = tf.get_variable("final_fc_weights", shape=[last_conv_element_count, label_count], initializer=tf.contrib.layers.xavier_initializer())
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattend_last_conv, final_fc_weights) + final_fc_bias

    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def bias_variable(shape, value=0.1):
    return tf.Variable(tf.constant(value, shape=shape))


def fire_module(layer_name, layer_input, s1x1, e1x1, e3x3, residual=False):
    """ Fire module consists of squeeze and expand convolutional layers. """
    fire = {}
    shape = layer_input.get_shape()
    # squeeze
    s1_weight = tf.get_variable(layer_name + '_s1', shape=[1, 1, int(shape[3]), s1x1], initializer=tf.contrib.layers.xavier_initializer())
    # expand
    e1_weight = tf.get_variable(layer_name + '_e1', shape=[1, 1, s1x1, e1x1], initializer=tf.contrib.layers.xavier_initializer())
    e3_weight = tf.get_variable(layer_name + '_e3', shape=[3, 3, s1x1, e3x3], initializer=tf.contrib.layers.xavier_initializer())

    fire['s1'] = tf.nn.conv2d(layer_input, s1_weight, strides=[1, 1, 1, 1], padding='SAME')
    fire['relu1'] = tf.nn.relu(fire['s1'] + bias_variable([s1x1]))

    fire['e1'] = tf.nn.conv2d(fire['relu1'], e1_weight, strides=[1, 1, 1, 1], padding='SAME')
    fire['e3'] = tf.nn.conv2d(fire['relu1'], e3_weight, strides=[1, 1, 1, 1], padding='SAME')

    fire['concat'] = tf.concat([tf.add(fire['e1'], bias_variable([e1x1])),
                                tf.add(fire['e3'], bias_variable([e3x3]))], 3)
    if residual:
        fire['relu2'] = tf.nn.relu(tf.add(fire['concat'], layer_input))
    else:
        fire['relu2'] = tf.nn.relu(fire['concat'])
    return fire['relu2']

## Regularizations
def BatchNorm(input, is_train, decay=0.999, name='BatchNorm'):
  '''
  https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/layers.py
  https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
  http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow
  '''
  from tensorflow.python.training import moving_averages

  axis = list(range(len(input.get_shape()) - 1))
  fdim = input.get_shape()[-1:]

  with tf.variable_scope(name):
    beta = tf.get_variable('beta', fdim, initializer=tf.constant_initializer(value=0.0))
    gamma = tf.get_variable('gamma', fdim, initializer=tf.constant_initializer(value=1.0))
    moving_mean = tf.get_variable('moving_mean', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)
    moving_variance = tf.get_variable('moving_variance', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)

    def mean_var_with_update():
      batch_mean, batch_variance = tf.nn.moments(input, axis)
      update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=True)
      update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay, zero_debias=True)
      with tf.control_dependencies([update_moving_mean, update_moving_variance]):
        return tf.identity(batch_mean), tf.identity(batch_variance)
    
    #mean, variance = control_flow_ops.cond(tf.cast(is_train, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))

    if is_train:
      mean, variance = mean_var_with_update()
    else:
      mean, variance = moving_mean, moving_variance

  return tf.nn.batch_normalization(input, mean, variance, beta, gamma, 1e-3)


def LeakyReLU(input, alpha=0.2):
  return tf.maximum(input, alpha*input)
