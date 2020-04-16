# ========================================================================
# Resize, Pad Image to Square Shape and Keep Its Aspect Ratio With Python
# ========================================================================

import os
import argparse
import sys

from PIL import Image, ImageOps

import tensorflow as tf


FLAGS = None


def main(_):

  target_dir = FLAGS.target_dir
  dir = os.path.dirname(target_dir)
  if not os.path.exists(dir):
    try:
      original_umask = os.umask(0)
      os.makedirs(dir, 0o775)
    finally:
      os.umask(original_umask)

  images = os.listdir(FLAGS.original_dir)
  # print(images)
  for image in images:
    imagepath = FLAGS.original_dir + image
    # tmp = os.path.splitext(image)
    im = Image.open(imagepath)
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(FLAGS.desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    # im = im.resize(new_size, Image.NEAREST)
    # im = im.resize(new_size, Image.BILINEAR)
    # im = im.resize(new_size, Image.BICUBIC)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (FLAGS.desired_size, FLAGS.desired_size))
    new_im.paste(im, ((FLAGS.desired_size - new_size[0]) // 2,
                      (FLAGS.desired_size - new_size[1]) // 2))
    new_im.save(target_dir + image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--original_dir',
    type=str,
    default='../../../dl_data/plant_seedlings/original_test/',
    help='Where is image to load.')
  parser.add_argument(
    '--target_dir',
    type=str,
    default='../../../dl_data/plant_seedlings/test/',
    help='Where is resized image to save.')
  parser.add_argument(
    '--desired_size',
    type=int,
    default=224,
    help='how do you want image resize height, width.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)