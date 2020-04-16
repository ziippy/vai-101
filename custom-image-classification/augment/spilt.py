import argparse
import os.path
import sys
import random
import shutil

import tensorflow as tf

FLAGS = None

def main(_):
  file_list = os.listdir(FLAGS.original_dir)

  for file in file_list:
    class_name = FLAGS.original_dir + file
    images = os.listdir(class_name)

    random.shuffle(images)

    total_size = len(images)
    training_size = round(total_size * 0.8)
    validation_size = round(total_size * 0.1)
    # testing_size = total_size - training_size - validation_size

    training = images[0:training_size]
    validation = images[training_size:training_size+validation_size]
    testing = images[-validation_size+1:]

    dest = './training/' + file + '/'
    dir = os.path.dirname(dest)
    if not os.path.exists(dir):
      os.makedirs(dir)
    for file_name in training:
      full_file_name = os.path.join(class_name, file_name)
      if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest + file_name)

    dest = './validation/' + file + '/'
    dir = os.path.dirname(dest)
    if not os.path.exists(dir):
      os.makedirs(dir)
    for file_name in validation:
      full_file_name = os.path.join(class_name, file_name)
      if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest + file_name)

    dest = './testing/' + file + '/'
    dir = os.path.dirname(dest)
    if not os.path.exists(dir):
      os.makedirs(dir)
    for file_name in testing:
      full_file_name = os.path.join(class_name, file_name)
      if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest + file_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--original_dir',
    type=str,
    default='../../../dl_data/DeepSight/resizing_from_normalize/',
    help='Where is image to load.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)