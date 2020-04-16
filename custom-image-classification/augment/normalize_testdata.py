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

  image_datagen = ImageDataGenerator(
                                      featurewise_center=True,
                                      # samplewise_center=True,
                                      featurewise_std_normalization=True,
                                      # samplewise_std_normalization=True,
                                      # rotation_range=40,
                                      # width_shift_range=0.2,
                                      # height_shift_range=0.2,
                                      # rescale=1./255,
                                      # shear_range=0.2,
                                      # zoom_range=0.5,
                                      # fill_mode='constant'
  )


  def modify_imagename():
    images = os.listdir(FLAGS.target_dir)
    for image in images:
      im = os.path.splitext(image)
      new = im[0].split('|')
      os.rename(FLAGS.target_dir + image, FLAGS.target_dir + new[0] + im[1])

  def normalize(original_path, target_dir, prefix):
    img = load_img(original_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, ?, ?)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, ?, ?)

    image_datagen.fit(x)

    batches = 1
    for batch in image_datagen.flow(x,
                                    batch_size=1,
                                    save_to_dir=target_dir,
                                    save_prefix=prefix):
      batches += 1
      if batches > 1:
        break  # otherwise the generator would loop indefinitely


  dir = os.path.dirname(FLAGS.target_dir)
  if not os.path.exists(dir):
    try:
      original_umask = os.umask(0)
      os.makedirs(dir, 0o775)
    finally:
      os.umask(original_umask)

  images = os.listdir(FLAGS.original_dir)
  for image in images:
    original_path = FLAGS.original_dir + image
    im = os.path.splitext(image)
    normalize(original_path, FLAGS.target_dir, im[0]+'|')

  modify_imagename()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--original_dir',
    type=str,
    default='../../../dl_data/plant_seedlings/test/',
    help='Where is image to load.')
  parser.add_argument(
    '--target_dir',
    type=str,
    default='../../../dl_data/plant_seedlings/norm_test/',
    help='Where is image to save.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)