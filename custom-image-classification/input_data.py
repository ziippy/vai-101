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

import hashlib
import os.path
import random
import re
import sys
import tarfile
import urllib

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
RANDOM_SEED = 59185


def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.
  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.
  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.
  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.
  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


class Data(object):
  def __init__(self, data_dir, labels,
               validation_percentage, testing_percentage):
    self.data_dir = data_dir
    # self.maybe_download_and_extract_dataset(data_url, data_dir)
    self._prepare_data_index(labels, validation_percentage, testing_percentage)


  def get_data(self, mode):
    return self.data_index[mode]


  def get_size(self, mode):
    """Calculates the number of samples in the dataset partition.
    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.
    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])


  def get_label_to_index(self):
    return self.label_to_index


  def _prepare_data_index(self, labels, validation_percentage, testing_percentage):
    """Prepares a list of the samples organized by set and label.
    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.
    Args:
      labels: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.
    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.
    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    labels_index = {}
    for index, label in enumerate(labels):
      labels_index[label] = index
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    all_labels = {}
    # Look through all the subfolders to find image samples
    search_path = os.path.join(self.data_dir, '*', '*')
    for image_path in gfile.Glob(search_path):
      _, label = os.path.split(os.path.dirname(image_path))
      # label = label.lower()
      all_labels[label] = True
      set_index = which_set(image_path, validation_percentage, testing_percentage)
      self.data_index[set_index].append({'label': label, 'file': image_path})
    if not all_labels:
      raise Exception('No image found at ' + search_path)
    for index, label in enumerate(labels):
      if label not in all_labels:
        raise Exception('Expected to find ' + label +
                        ' in labels but only found ' +
                        ', '.join(all_labels.keys()))

    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])

    # Prepare the rest of the result data structure.
    self.labels_list = labels
    self.label_to_index = {}
    for label in all_labels:
      if label in labels_index:
        self.label_to_index[label] = labels_index[label]


  def maybe_download_and_extract_dataset(self, data_url, dest_directory):
    """Download and extract data set tar file.
    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.
    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url:
      return
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        tf.logging.error('Failed to download URL: %s to folder: %s', data_url,
                         filepath)
        tf.logging.error('Please make sure you have enough free space and'
                         ' an internet connection')
        raise
      print()
      statinfo = os.stat(filepath)
      tf.logging.info('Successfully downloaded %s (%d bytes)', filename,
                      statinfo.st_size)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


  # def _shuffle_data(self):
  #   training_size = self.get_size('training')
  #
  #   tmp_data = self.data_index['training'].copy()
  #   tmp_data.extend(self.data_index['validation'].copy())
  #
  #   random.shuffle(tmp_data)
  #
  #   self.data_index['training'] = None
  #   self.data_index['validation'] = None
  #
  #   self.data_index['training'] = tmp_data[:training_size]
  #   self.data_index['validation'] = tmp_data[training_size:]



class ImageDataGenerator(object):
  # class ImageDataGenerator(object):
  """
  Wrapper class around the new Tensorflows dataset pipeline.

  Handles loading, partitioning, and preparing training data.
  Requires Tensorflow >= version 1.12rc0
  """

  def __init__(self, data, label_to_index, batch_size, shuffle=True):
    # if shuffle:
    #   self._shuffle_data() # initial shuffling

    self.data_size = len(data)

    # shuffle in advance
    images, labels = self._get_data(data, label_to_index)

    # create dataset, Creating a source
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # shuffle the first `buffer_size` elements of the dataset
    #  Make sure to call tf.data.Dataset.shuffle() before applying the heavy transformations
    # (like reading the images, processing them, batching...).
    if shuffle:
      dataset = dataset.shuffle(buffer_size= 100 * batch_size)

    # distinguish between train/infer. when calling the parsing functions
    # transform to images, preprocess, repeat, batch...
    dataset = dataset.map(self._parse_function, num_parallel_calls=8)

    dataset = dataset.prefetch(buffer_size = 10 * batch_size)

    # create a new dataset with batches of images
    dataset = dataset.batch(batch_size)

    self.dataset = dataset


  def _get_data(self, data, label_to_index):
    sample_count = len(data)
    # Data and labels will be populated and returned.
    image_paths = np.zeros(sample_count, dtype="U200")
    labels = np.zeros(sample_count)

    for index in range(sample_count):
      sample = data[index]
      image_paths[index] = sample['file']
      labels[index] = label_to_index[sample['label']]

    # convert lists to TF tensor
    image_paths = convert_to_tensor(image_paths, dtype=dtypes.string)
    labels = convert_to_tensor(labels, dtype=dtypes.float64)

    return image_paths, labels


  def _parse_function(self, filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, label