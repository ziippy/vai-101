import numpy as np
import os
import argparse
import os.path
import sys

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

np.random.seed(888)

FLAGS = None


def main(_):

  # will modify like belows
  # https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn_tfaugment2d.py

  image_datagen = ImageDataGenerator( # featurewise_center=True,
                                      # samplewise_center=True,
                                      # featurewise_std_normalization=True,
                                      # samplewise_std_normalization=True,
                                      rotation_range=50,
                                      width_shift_range=0.3,
                                      height_shift_range=0.3,
                                      rescale=1./255,
                                      shear_range=0.3,
                                      zoom_range=0.5,
                                      fill_mode='constant')


  def maxnum_of_class_images(file_list):
    nums_per_class = []

    for classname in file_list:
      class_path = FLAGS.original_dir + str(classname)
      images = os.listdir(class_path)
      nums_per_class.append(len(images))

    nums_per_class.sort(reverse=True)

    return nums_per_class[0] * FLAGS.augment_size


  def augment(original_path, target_dir, prefix, augnum):
    img = load_img(original_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    batches = 1
    for batch in image_datagen.flow(x,
                                    batch_size=1,
                                    save_to_dir=target_dir,
                                    save_prefix=prefix):
      batches += 1
      if batches > augnum:
        break  # otherwise the generator would loop indefinitely

  file_list = os.listdir(FLAGS.original_dir)
  file_list.sort()

  maxnum = maxnum_of_class_images(file_list)

  for classname in file_list:
    class_path = FLAGS.original_dir + str(classname)
    images = os.listdir(class_path)
    augnum = maxnum // len(images)

    target_class_dir = FLAGS.target_dir + str(classname) + '/'
    dir = os.path.dirname(target_class_dir)
    if not os.path.exists(dir):
      try:
        original_umask = os.umask(0)
        os.makedirs(dir, 0o775)
      finally:
        os.umask(original_umask)

    for image in images:
      original_path = class_path + '/' + image
      im = os.path.splitext(image)
      augment(original_path, target_class_dir, im[0], augnum)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--original_dir',
    type=str,
    default='../../../dl_data/plant_seedlings/train/',
    help='Where is image to load.')
  parser.add_argument(
    '--target_dir',
    type=str,
    default='../../../dl_data/plant_seedlings/train/',
    help='Where is image to save.')
  parser.add_argument(
    '--augment_size',
    type=int,
    default=12,
    help='How many items to augment with at once')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)