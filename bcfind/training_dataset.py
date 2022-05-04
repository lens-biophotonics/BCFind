import logging
import numpy as np
import tensorflow as tf

from pathlib import Path

from zetastitcher import InputFile
from bcfind.make_training_data import get_target
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
        return input_image, blobs

    def __new__(cls, tiff_list, marker_list, batch_size, dim_resolution=1.0, output_shape=None, augmentations=None, augmentations_prob=0.5):
        if isinstance(dim_resolution, (float, int)):
            dim_resolution = [dim_resolution] * 3

        # with tf.device('/cpu:0'):
        data = tf.data.Dataset.from_tensor_slices((tiff_list, marker_list))

        # load images and targets from paths
        data = data.map(lambda x, y: cls.parse_imgs(x, y, dim_resolution), num_parallel_calls=tf.data.AUTOTUNE)

        # cache data after time consuming map and make it shuffle every epoch
        data = data.cache().shuffle(len(marker_list), reshuffle_each_iteration=True)

        # crop inputs and targets
        if output_shape is not None:
            data = data.map(lambda x, y: random_crop_tf(x, y, output_shape), num_parallel_calls=tf.data.AUTOTUNE)

        # do augmentations
        if augmentations is not None:
            ops_list = get_op_list(augmentations)
            data = data.map(lambda x, y: augment(x, y, ops_list, augmentations_prob), num_parallel_calls=tf.data.AUTOTUNE)
        
        # add channel dimension
        data = data.map(lambda x, y: (tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)), num_parallel_calls=tf.data.AUTOTUNE)
        
        data = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data


if __name__ == '__main__':
    import os
    import time
    
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[-1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[-1], True)
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # marker_dir = '/mnt/NASone/curzio/Data/I48/Gray_matter/GT_files/Train'
    # img_dir = '/mnt/NASone/curzio/Data/I48/Gray_matter/Tiff_files/Train'

    # marker_files = [f'{marker_dir}/{f}' for f in os.listdir(marker_dir)]
    # img_files = [f'{img_dir}/{f}' for f in os.listdir(img_dir)]

    data_dir = '/mnt/NASone/curzio/Data/SST'
    marker_files = [f'{data_dir}/GT_files/Train/{fname}' for fname in os.listdir(f'{data_dir}/GT_files/Train')]
    img_files = [f'{data_dir}/Tiff_files/Train/{fname}' for fname in os.listdir(f'{data_dir}/Tiff_files/Train')]
    
    sorted_tiff_list = []
    for f in marker_files:
        fname = Path(f).with_suffix('').name
        tiff_file = [f for f in map(lambda f: Path(f), img_files) if f.name == fname]
        sorted_tiff_list.append(str(tiff_file[0]))

    data = TrainingDataset(
        sorted_tiff_list[:],
        marker_files[:],
        batch_size=4,
        output_shape=[80, 240, 240],
        dim_resolution=[2.0, 0.65, 0.65],
        augmentations={
            'gamma': [0.5, 2.0],
            'rotation': [-180, 180],
            'zoom': [1.3, 1.5],
            'brightness': [-0.3, 0.3],
            'noise': [0.001, 0.05],
            'blur': [0.01, 1.5]
            # 'myfunc': lambda x, y: (x - tf.reduce_min(x), y),

            },
        augmentations_prob=1.0,
        )

    for epoch in range(5):
        s = time.time()
        for x, y in data:
            x_shape = x.shape
            x_min = tf.reduce_min(x, axis=[1, 2, 3, 4]).numpy()
            y_shape = y.shape
            y_min = tf.reduce_min(y, axis=[1, 2, 3, 4]).numpy()
        e = time.time()
        print()
        print(f'Time elapsed for epoch {epoch + 1}: {np.round(e-s, 4)} seconds.')
        print()