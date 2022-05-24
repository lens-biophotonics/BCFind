import logging
import numpy as np
import tensorflow as tf

from pathlib import Path

from zetastitcher import InputFile
from bcfind.artificial_targets import get_target
from bcfind.augmentation import *


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@tf.function
def get_target_tf(marker_file, target_shape, dim_resolution):
    def get_target_wrap(marker_file, target_shape, dim_resolution):
        marker_file = Path(marker_file.decode())
        blobs = get_target(
            marker_file,
            target_shape=target_shape, 
            default_radius=3.5,  # FIXME: not yet configurable!!
            dim_resolution=dim_resolution,
        )
        return blobs.astype(np.float32)

    target = tf.numpy_function(get_target_wrap, [marker_file, target_shape, dim_resolution], tf.float32)
    return target

@tf.function
def get_input_tf(input_file):
    def get_input_wrap(input_file):
        input_file = Path(input_file.decode())
        input_image = InputFile(input_file).whole()
        return input_image.astype(np.float32)

    input = tf.numpy_function(get_input_wrap, [input_file], tf.float32)
    return input

@tf.function
def normalize_tf(x):
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    return (x - x_min) / (x_max - x_min)


class TrainingDataset(tf.data.Dataset):
    """ Training data flow for 3D gray images with cell locations as targets.

    Args:
        tiff_list (list):

        marker_list (list):

        batch_size (list, tuple):

        output_shape (list, tuple):

        dim_resolution (float, list, tuple):

        augmentations (optional, list, dict):

        augmentations_prob (optional, float, list):

    Returns:
        tensorflow.datasets.Data: a tensorflow dataset.

    Yields:
        tensorflow.Tensor: batch of inputs and targets.
    """
    @staticmethod
    @tf.function
    def parse_imgs(tiff_path, marker_path, dim_resolution):
        logger.info(f'loading {tiff_path}')
        input_image = get_input_tf(tiff_path)
        input_image = normalize_tf(input_image)
        shape = tf.shape(input_image)

        logger.info(f'creating blobs from {marker_path}')
        shape_T = [shape[d] for d in (2, 1, 0)]
        dim_resolution_T = [dim_resolution[d] for d in (2, 1, 0)]
        blobs = get_target_tf(marker_path, shape_T, dim_resolution_T)
        blobs = tf.transpose(blobs, perm=(2, 1, 0))

        xy = tf.concat([tf.expand_dims(input_image, 0), tf.expand_dims(blobs, 0)], axis=0)
        return tf.ensure_shape(xy, (2, None, None, None))

    def __new__(cls, tiff_list, marker_list, batch_size, dim_resolution=1.0, output_shape=None, augmentations=None, augmentations_prob=0.5):
        if isinstance(dim_resolution, (float, int)):
            dim_resolution = [dim_resolution] * 3

        # with tf.device('/cpu:0'):
        data = tf.data.Dataset.from_tensor_slices((tiff_list, marker_list))
        
        # load images and targets from paths
        data = data.map(lambda x, y: cls.parse_imgs(x, y, dim_resolution), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        # cache data after time consuming map and make it shuffle every epoch
        data = data.cache().shuffle(len(marker_list), reshuffle_each_iteration=True)

        # crop inputs and targets
        if output_shape is not None:
            data = data.map(lambda xy: random_crop_tf(xy, output_shape), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        
        # do augmentations
        if augmentations is not None:
            data = data.map(lambda xy: augment(xy, augmentations, augmentations_prob), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        
        # add channel dimension
        data = data.map(lambda xy: tf.expand_dims(xy, axis=-1), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        
        # unstack xy
        data = data.map(lambda xy: tf.unstack(xy), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        
        return data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
