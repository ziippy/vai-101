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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import csv

from tqdm import tqdm

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import models
# import input_data
import prediction_data
from tensorflow.python.platform import gfile

from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  label_count = 10


  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_epochs=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_epochs_list = list(map(int, FLAGS.how_many_training_epochs.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_epochs_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_epochs and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_epochs_list),
                                                   len(learning_rates_list)))

  input_xs = tf.placeholder(tf.float32, [None, 784], name='input_xs')
  logits, dropout_prob = models.create_model(
      input_xs,
      label_count,
      FLAGS.model_architecture,
      is_training=True)

  # Define loss and optimizer
  ground_truth_input = tf.placeholder(tf.float32, [None, 10], name='groundtruth_input')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.softmax_cross_entropy(
      onehot_labels=ground_truth_input, logits=logits)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)
  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
    momentum = tf.placeholder(tf.float32, [], name='momentum')
    # train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(cross_entropy_mean)
    train_step = tf.train.MomentumOptimizer(learning_rate_input, momentum, use_nesterov=True).minimize(cross_entropy_mean)
    # train_step = tf.train.AdamOptimizer(learning_rate_input).minimize(cross_entropy_mean)
    # train_step = tf.train.AdadeltaOptimizer(learning_rate_input).minimize(cross_entropy_mean)
    # train_step = tf.train.RMSPropOptimizer(learning_rate_input, momentum).minimize(cross_entropy_mean)

  predicted_indices = tf.argmax(logits, 1)
  correct_prediction = tf.equal(predicted_indices, tf.argmax(ground_truth_input,1))
  confusion_matrix = tf.confusion_matrix(
    tf.argmax(ground_truth_input,1), predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)

  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  tf.global_variables_initializer().run()

  start_epoch = 1
  start_checkpoint_epoch = 0
  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    tmp = FLAGS.start_checkpoint
    tmp = tmp.split('-')
    tmp.reverse()
    start_checkpoint_epoch = int(tmp[0])
    start_epoch = start_checkpoint_epoch + 1

  # calculate training epochs max
  training_epochs_max = np.sum(training_epochs_list)

  # start_checkpoint 값과 training_epochs_max 값이 다를 경우에만 training 수행
  if start_checkpoint_epoch != training_epochs_max:
    tf.logging.info('Training from epoch: %d ', start_epoch)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pbtxt')


  # Training epoch
  for training_epoch in xrange(start_epoch, training_epochs_max + 1):
    # Figure out what the current learning rate is.
    training_epochs_sum = 0
    for i in range(len(training_epochs_list)):
      training_epochs_sum += training_epochs_list[i]
      if training_epoch <= training_epochs_sum:
        learning_rate_value = learning_rates_list[i]
        break

    set_size = mnist.train.num_examples
    for i in xrange(0, set_size, FLAGS.batch_size):
      # Pull the image samples we'll use for training.
      train_batch_xs, train_batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      # Run the graph with this batch of training data.
      train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
          [
              merged_summaries, evaluation_step, cross_entropy_mean, train_step,
              increment_global_step
          ],
          feed_dict={
              input_xs: train_batch_xs,
              ground_truth_input: train_batch_ys,
              learning_rate_input: learning_rate_value,
              momentum: 0.95,
              dropout_prob: 0.5
          })
      train_writer.add_summary(train_summary, i)
      tf.logging.info('Epoch #%d, Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                      (training_epoch, i, learning_rate_value, train_accuracy * 100,
                       cross_entropy_value))

      is_last_step = ((set_size - i) / FLAGS.batch_size <= 1)
      if is_last_step:
        set_size = mnist.validation.num_examples
        total_accuracy = 0
        total_conf_matrix = None
        for i in xrange(0, set_size, FLAGS.batch_size):
          validation_batch_xs, validation_batch_ys = \
            mnist.validation.next_batch(FLAGS.batch_size)
          # Run a validation step and capture training summaries for TensorBoard
          # with the `merged` op.
          validation_summary, validation_accuracy, conf_matrix = sess.run(
              [merged_summaries, evaluation_step, confusion_matrix],
              feed_dict={
                  input_xs: validation_batch_xs,
                  ground_truth_input: validation_batch_ys,
                  dropout_prob: 1.0
              })
          validation_writer.add_summary(validation_summary, training_epoch)
          batch_size = min(FLAGS.batch_size, set_size - i)
          total_accuracy += (validation_accuracy * batch_size) / set_size
          if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
          else:
            total_conf_matrix += conf_matrix

        tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        tf.logging.info('Epoch %d: Validation accuracy = %.1f%% (N=%d)' %
                        (training_epoch, total_accuracy * 100, set_size))


    # Save the model checkpoint periodically.
    if (training_epoch % FLAGS.save_step_interval == 0 or
        training_epoch == training_epochs_max):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     FLAGS.model_architecture + '.ckpt')
      tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_epoch)
      saver.save(sess, checkpoint_path, global_step=training_epoch)

  # For testing
  set_size = mnist.test.num_examples
  tf.logging.info('test size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_batch_xs, test_batch_ys = mnist.test.next_batch(FLAGS.batch_size)
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            input_xs: test_batch_xs,
            ground_truth_input: test_batch_ys,
            dropout_prob: 1.0
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix

  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                           set_size))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--how_many_training_epochs',
      type=str,
      default='4,3',
      help='How many training loops to run')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='./models/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--train_dir',
      type=str,
      default='./models',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=1,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='mnist_mobile',
      help='What model architecture to use')
  parser.add_argument(
      '--prediction_batch_size',
      type=int,
      default=100,
      help='How many items to predict with at once', )
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)