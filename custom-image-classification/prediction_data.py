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

import os.path
import random

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

RANDOM_SEED = 59185


class Data(object):
  def __init__(self, prediction_data_dir):
    self.prediction_data_dir = prediction_data_dir
    self._prepare_data_index()


  def get_data(self):
    return self.data_index['prediction']


  def get_size(self):
    return len(self.data_index['prediction'])


  def _prepare_data_index(self):
    random.seed(RANDOM_SEED)

    self.data_index = {'prediction': []}
    # Look through all the subfolders to find audio samples
    search_path = os.path.join(self.prediction_data_dir, '*')
    for image_path in gfile.Glob(search_path):
      self.data_index['prediction'].append({'file': image_path})


class ImageDataGenerator(object):
  """
  Wrapper class around the new Tensorflows dataset pipeline.

  Handles loading, partitioning, and preparing training data.
  Requires Tensorflow >= version 1.12rc0
  """

  def __init__(self, data, batch_size):
    # if shuffle:
    #   self._shuffle_data() # initial shuffling

    self.data_size = len(data)

    images_path, images_name = self._get_data(data)

    # create dataset, Creating a source
    dataset = tf.data.Dataset.from_tensor_slices((images_path, images_name))

    # distinguish between train/infer. when calling the parsing functions
    # transform to images, preprocess, repeat, batch...
    dataset = dataset.map(self._parse_function, num_parallel_calls=8)

    dataset = dataset.prefetch(buffer_size = 10 * batch_size)

    # create a new dataset with batches of images
    dataset = dataset.batch(batch_size)

    self.dataset = dataset


  def _get_data(self, data):
    sample_count = len(data)
    # Data will be populated and returned.
    image_paths = np.zeros(sample_count, dtype="U200")
    image_names = np.empty(sample_count, dtype="U50")

    for index in range(sample_count):
      sample = data[index]
      image_paths[index] = sample['file']
      image_names[index] = os.path.basename(sample['file'])

    # convert lists to TF tensor
    image_paths = convert_to_tensor(image_paths, dtype=dtypes.string)
    image_names = convert_to_tensor(image_names, dtype=dtypes.string)

    return image_paths, image_names


  def _parse_function(self, image_path, image_name):
    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, image_name

