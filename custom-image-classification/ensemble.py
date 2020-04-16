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
import datetime

from tqdm import tqdm

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import model as models
import prediction_data

from input_data import Data
from input_data import ImageDataGenerator

from tensorflow.python.platform import gfile

import os

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

FLAGS = None

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

    labels = FLAGS.labels.split(',')
    label_count = len(labels)

    training_epochs_list = list(map(int, FLAGS.how_many_training_epochs.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_epochs_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_epochs and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_epochs_list),
                                                       len(learning_rates_list)))

    input_xs = tf.placeholder(
        tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='input_xs')

    logits, dropout_prob = models.create_model(
        input_xs,
        label_count,
        FLAGS.model_architecture,
        is_training=True)

    # Define loss and optimizer
    ground_truth_input = tf.placeholder(tf.int64, [None], name='groundtruth_input')

    # Optionally we can add runtime checks to spot when NaNs or other symptoms of
    # numerical errors start occurring during training.
    control_dependencies = []
    if FLAGS.check_nans:
        checks = tf.add_check_numerics_ops()
        control_dependencies = [checks]

    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=logits)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
        learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
        momentum = tf.placeholder(tf.float32, [], name='momentum')
        # train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(cross_entropy_mean)
        # train_step = tf.train.MomentumOptimizer(learning_rate_input, momentum, use_nesterov=True).minimize(cross_entropy_mean)
        # train_step = tf.train.AdamOptimizer(learning_rate_input).minimize(cross_entropy_mean)
        # train_step = tf.train.AdadeltaOptimizer(learning_rate_input).minimize(cross_entropy_mean)
        train_step = tf.train.RMSPropOptimizer(learning_rate_input, momentum).minimize(cross_entropy_mean)

    predicted_indices = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predicted_indices, ground_truth_input)
    confusion_matrix = tf.confusion_matrix(
        ground_truth_input, predicted_indices, num_classes=label_count)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    saver = tf.train.Saver(tf.global_variables())

    merged_summaries = tf.summary.merge_all()

    tf.global_variables_initializer().run()


    ############################
    # start prediction
    ############################
    print("{} Start prediction".format(datetime.datetime.now()))

    id2name = {i: name for i, name in enumerate(labels)}
    submission = dict()

    # Place data loading and preprocessing on the cpu
    raw_data2 = prediction_data.Data(FLAGS.prediction_data_dir)
    pre_data = prediction_data.ImageDataGenerator(raw_data2.get_data(),
                                                  FLAGS.prediction_batch_size)

    # create an reinitializable iterator given the dataset structure
    iterator = tf.data.Iterator.from_structure(pre_data.dataset.output_types,
                                               pre_data.dataset.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    prediction_init_op = iterator.make_initializer(pre_data.dataset)

    # Get the number of training/validation steps per epoch
    pre_batches_per_epoch = int(np.floor(pre_data.data_size / FLAGS.prediction_batch_size)) + 1

    print("Test Size : {}".format(raw_data2.get_size()))

    count = 0;
    sess.run(prediction_init_op)
    ckpt_list = FLAGS.ckpt_list.split(',')
    ckpt_size = len(ckpt_list)

    for i in range(pre_batches_per_epoch):

        pred_labels = []
        pred_xs, fnames = sess.run(next_batch)

        for j in range(ckpt_size):
          models.load_variables_from_checkpoint(sess, ckpt_list[j])

          prediction, predicted_label = sess.run([predicted_indices, logits],
                                feed_dict={
                                    input_xs: pred_xs,
                                    dropout_prob: 1.0
                                })

          pred_prob = tf.nn.softmax(predicted_label)
          pred_labels.append(sess.run(pred_prob))

        pred_label_array = np.array(pred_labels)
        ensemble_pred_labels = np.mean(pred_label_array, axis = 0)
        ensemble_class_pred = np.argmax(ensemble_pred_labels, axis = 1)

        size = len(fnames)
        for n in xrange(0, size):
            submission[fnames[n].decode('UTF-8')] = id2name[ensemble_class_pred[n]]

        count += size
        print(count, ' completed')

    # make submission.csv
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)

    fout = open(os.path.join(FLAGS.result_dir,
                             'submission_' + FLAGS.model_architecture + '_ensemble_' +
                             FLAGS.how_many_training_epochs + '.csv'),
                'w', encoding='utf-8', newline='')
    writer = csv.writer(fout)
    writer.writerow(['file', 'species'])
    for key in sorted(submission.keys()):
        writer.writerow([key, submission[key]])
    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--data_url',
    #     type=str,
    #     # pylint: disable=line-too-long
    #     default='http://download.tensorflow.org/data/image_commands_v0.01.tar.gz',
    #     # pylint: enable=line-too-long
    #     help='Location of image training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../../dl_data/plant_seedlings/train/',
        # default='../dataset/training/',
        help="""\
        Where to download the image training data to.
        """)
    parser.add_argument(
        '--prediction_data_dir',
        type=str,
      default='../../dl_data/plant_seedlings/test/',
        # default='../test/',
        help="""\
        Where is image prediction data.
        """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--how_many_training_epochs',
        type=str,
        default='2',
        help='How many training loops to run')
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=1,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.0001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='How many items to train with at once', )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='./models/retrain_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--labels',
        type=str,
        default='Black-grass,Charlock,Cleavers,Common Chickweed,Common wheat,Fat Hen,Loose Silky-bent,Maize,Scentless Mayweed,Shepherds Purse,Small-flowered Cranesbill,Sugar beet',
        help='Labels to use', )
    parser.add_argument(
        '--ckpt_list',
        type=str,
        default='./models/mobile.ckpt-15,./models/mobile_111.ckpt-9',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='mobile',
        help='What model architecture to use')
    parser.add_argument(
        '--prediction_batch_size',
        type=int,
        default=200,
        help='How many items to predict with at once', )
    parser.add_argument(
        '--result_dir',
        type=str,
        default='./result',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
