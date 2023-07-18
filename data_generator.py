import numpy as np
import cv2
import skimage
import yaml
import tensorflow as tf

from helper import get_partial_df, get_config

config = get_config()

img_width = int(config["img_width"])
img_height = int(config["img_height"])
batch_size = int(config["batch_size"])


# Train data generator


def train_generator(train_dir, train_dir_gt):
    while True:
        for start in range(0, len(train_dir), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(train_dir))
            for img_path in range(start, end):
                img = cv2.imread(train_dir[img_path])

                img = skimage.transform.resize(
                    img,
                    (img_height, img_width),
                    preserve_range=True,
                    anti_aliasing=False,
                    order=0,
                )

                x_batch.append(img)

                img = cv2.imread(train_dir_gt[img_path])
                img = skimage.transform.resize(
                    img,
                    (img_height, img_width, 1),
                    preserve_range=True,
                    anti_aliasing=False,
                    order=0,
                )

                y_batch.append(img)

            y_batch = tf.keras.utils.to_categorical(y_batch, num_classes=5)

            yield (np.array(x_batch), np.array(y_batch))


def valid_generator(valid_dir, valid_dir_gt):
    while True:
        for start in range(0, len(valid_dir), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(valid_dir))
            for img_path in range(start, end):
                img = cv2.imread(valid_dir[img_path])

                img = skimage.transform.resize(
                    img,
                    (img_height, img_width),
                    preserve_range=True,
                    anti_aliasing=False,
                    order=0,
                )

                x_batch.append(img)

                img = cv2.imread(valid_dir_gt[img_path])
                img = skimage.transform.resize(
                    img,
                    (img_height, img_width, 1),
                    preserve_range=True,
                    anti_aliasing=False,
                    order=0,
                )

                y_batch.append(img)

            y_batch = tf.keras.utils.to_categorical(y_batch, num_classes=5)

            yield (np.array(x_batch), np.array(y_batch))
