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

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import model
import prediction_data

from input_data import Data
from input_data import ImageDataGenerator

from tensorflow.python.platform import gfile

FLAGS = None


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

    labels = FLAGS.labels.split(',')
    label_count = len(labels)

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        raw_data = Data(FLAGS.data_dir,
                       labels,
                       FLAGS.validation_percentage,
                       FLAGS.testing_percentage)

        tr_data = ImageDataGenerator(raw_data.get_data('training'),
                                     raw_data.get_label_to_index(),
                                     FLAGS.batch_size)

        val_data = ImageDataGenerator(raw_data.get_data('validation'),
                                      raw_data.get_label_to_index(),
                                      FLAGS.batch_size)

        te_data = ImageDataGenerator(raw_data.get_data('testing'),
                                     raw_data.get_label_to_index(),
                                     FLAGS.batch_size)

        # create an reinitializable iterator given the dataset structure
        iterator = tf.data.Iterator.from_structure(tr_data.dataset.output_types,
                                           tr_data.dataset.output_shapes)
        next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.dataset)
    validation_init_op = iterator.make_initializer(val_data.dataset)
    testing_init_op = iterator.make_initializer(te_data.dataset)


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

    input_xs = tf.placeholder(
        tf.float32, [None, FLAGS.image_hw, FLAGS.image_hw, 3], name='input_xs')
    logits, dropout_prob = model.create_model(
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

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    tf.global_variables_initializer().run()

    start_epoch = 1
    start_checkpoint_epoch = 0
    if FLAGS.start_checkpoint:
        model.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        tmp = FLAGS.start_checkpoint
        tmp = tmp.split('-')
        tmp.reverse()
        start_checkpoint_epoch = int(tmp[0])
        start_epoch = start_checkpoint_epoch + 1

    # calculate training epochs max
    training_epochs_max = np.sum(training_epochs_list)

    if start_checkpoint_epoch != training_epochs_max:
        tf.logging.info('Training from epoch: %d ', start_epoch)

    # Saving as Protocol Buffer (pb)
    # tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
    #                      FLAGS.model_architecture + '.pbtxt')
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pb',
                         as_text=False)

    # Save list of words.
    with gfile.GFile(
            os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
            'w') as f:
        f.write('\n'.join(raw_data.labels_list))

    # Get the number of training/validation steps per epoch
    tr_batches_per_epoch = int(tr_data.data_size / FLAGS.batch_size)
    if tr_data.data_size % FLAGS.batch_size > 0:
        tr_batches_per_epoch += 1
    val_batches_per_epoch = int(val_data.data_size / FLAGS.batch_size)
    if val_data.data_size % FLAGS.batch_size > 0:
        val_batches_per_epoch += 1
    te_batches_per_epoch = int(te_data.data_size / FLAGS.batch_size)
    if te_data.data_size % FLAGS.batch_size > 0:
        te_batches_per_epoch += 1


    ############################
    # Training loop.
    ############################
    for training_epoch in xrange(start_epoch, training_epochs_max + 1):
        # Figure out what the current learning rate is.
        training_epochs_sum = 0
        for i in range(len(training_epochs_list)):
            training_epochs_sum += training_epochs_list[i]
            if training_epoch <= training_epochs_sum:
                learning_rate_value = learning_rates_list[i]
                break

        # Initialize iterator with the training dataset
        sess.run(training_init_op)
        for step in range(tr_batches_per_epoch):
            # Pull the image samples we'll use for training.
            train_batch_xs, train_batch_ys = sess.run(next_batch)
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

            train_writer.add_summary(train_summary, step)
            tf.logging.info('Epoch #%d, Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                            (training_epoch, step, learning_rate_value, train_accuracy * 100,
                             cross_entropy_value))


        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.datetime.now()))
        # Reinitialize iterator with the validation dataset
        sess.run(validation_init_op)
        total_val_accuracy = 0
        validation_count = 0
        total_conf_matrix = None
        for i in range(val_batches_per_epoch):
            validation_batch_xs, validation_batch_ys = sess.run(next_batch)
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

            total_val_accuracy += validation_accuracy
            validation_count += 1
            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                total_conf_matrix += conf_matrix

        total_val_accuracy /= validation_count

        tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                        (training_epoch, total_val_accuracy * 100, raw_data.get_size('validation')))

        # Save the model checkpoint periodically.
        if (training_epoch % FLAGS.save_step_interval == 0 or
                training_epoch == training_epochs_max):
            checkpoint_path = os.path.join(FLAGS.train_dir,
                                           FLAGS.model_architecture + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_epoch)
            saver.save(sess, checkpoint_path, global_step=training_epoch)


    ############################
    # For Evaluate
    ############################
    start = datetime.datetime.now()
    print("{} Start testing".format(start))
    # Reinitialize iterator with the Evaluate dataset
    sess.run(testing_init_op)

    total_test_accuracy = 0
    test_count = 0
    total_conf_matrix = None
    for i in range(te_batches_per_epoch):
        test_batch_xs, test_batch_ys = sess.run(next_batch)
        test_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                input_xs: test_batch_xs,
                ground_truth_input: test_batch_ys,
                dropout_prob: 1.0
            })

        total_test_accuracy += test_accuracy
        test_count += 1

        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix

    total_test_accuracy /= test_count

    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_test_accuracy * 100,
                                                             raw_data.get_size('testing')))

    end = datetime.datetime.now()
    print('End testing: ', end)
    print('total testing time: ', end - start)


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
    pre_batches_per_epoch = int(pre_data.data_size / FLAGS.prediction_batch_size)
    if pre_data.data_size % FLAGS.prediction_batch_size > 0:
        pre_batches_per_epoch += 1

    count = 0;
    # Initialize iterator with the prediction dataset
    sess.run(prediction_init_op)
    for i in range(pre_batches_per_epoch):
        fingerprints, fnames = sess.run(next_batch)
        prediction = sess.run([predicted_indices],
                              feed_dict={
                                  input_xs: fingerprints,
                                  dropout_prob: 1.0
                              })
        size = len(fnames)
        for n in xrange(0, size):
            submission[fnames[n].decode('UTF-8')] = id2name[prediction[0][n]]

        count += size
        print(count, ' completed')

    # make submission.csv
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)

    fout = open(os.path.join(FLAGS.result_dir,
                             'submission_' + FLAGS.model_architecture + '_' +
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
        default='7,10',
        help='How many training loops to run')
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=1,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.001, 0.0001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
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
        '--train_dir',
        type=str,
        default='./models',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--result_dir',
        type=str,
        default='./result',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=1,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        # default='./models/mobile.ckpt-1',
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='mobile',
        help='What model architecture to use')
    parser.add_argument(
        '--prediction_batch_size',
        type=int,
        default=128,
        help='How many items to predict with at once', )
    parser.add_argument(
        '--image_hw',
        type=int,
        default=224,    # nasnet, mobilenet
        help='how do you want image resize height, width.')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
