import logging
from pathlib import Path

import numpy as np

import tensorflow as tf

from zetastitcher import InputFile

from bcfind.make_training_data import get_target
from bcfind.augmentation import random_crop, augment


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TrainingDataset:
    @staticmethod
    def _img_generator(marker_list):
        for marker_file in map(lambda x: Path(x.decode()), marker_list):
            logger.info(f'loading {marker_file.with_suffix("")}')
            input_image = InputFile(marker_file.with_suffix('')).whole().astype(np.float32)
            logger.info('creating blobs')
            blobs = get_target(marker_file, input_image.shape)

            yield input_image, blobs

    def __new__(cls, marker_list):
        def _my_gen():
            for el in crop_gen:
                yield el[0], el[1]

        img_gen = tf.data.Dataset.from_generator(
            cls._img_generator,
            output_signature=tf.TensorSpec(None, tf.float32),
            args=(marker_list,))

        crop_gen = img_gen \
            .cache() \
            .shuffle(len(marker_list), reshuffle_each_iteration=True) \
            .map(lambda x: tf.numpy_function(random_crop, [x], tf.float32))

        block_gen = tf.data.Dataset.from_generator(_my_gen, output_signature=(
                (tf.TensorSpec(([80, 240, 240]), tf.float32),) * 2))
        return block_gen.map(lambda x, y: (augment(x), y))
