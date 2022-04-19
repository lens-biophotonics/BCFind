import logging
import numpy as np
import functools as ft
import tensorflow as tf
from pathlib import Path

from zetastitcher import InputFile

from bcfind.make_training_data import get_target
from bcfind.augmentation import *


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_op_list(augmentations):
    """ Returns a list of callable operations from a list or dictionary of default or custom augmentations.

    Args:
        augmentations (list, tuple, dict): list or tuple of strings/callables or both. String elements will use default values of implemented operations.
                                    Strings must be one of [\'brightness\', \'contrast\', \'gamma\', \'noise\']. 
                                    Callable elements will be called, better if they are tensorflow.functions.
                                      dict of lists/callables or both. List values will be the parameter range of implemented operations. 
                                    Keys of list values must be on of [\'brightness\', \'contrast\', \'gamma\', \'noise\'].
                                    Callable values will be called, better if they are tensorflow.operations.
                                    Keys of callable values must be different from implemented operation names.
                                    Callables must take, in either cases, the tensor of an input image and return its augmented version.
                                    
    Raises:
        ValueError: if args are bad specified.

    Returns:
        list: list of tensorflow callable operations.
    """
    implemented_ops = {
            'gamma': random_gamma_tf,
            'contrast': random_contrast_tf,
            'brightness': random_brightness_tf,
            'noise': random_noise_tf,
            'rotation': random_rotation_tf,
            'zoom': random_zoom_tf,
            }
    
    ops_list = []
    if isinstance(augmentations, (list, tuple)):
        for op in augmentations:
            if isinstance(op, str):
                assert op in implemented_ops, f'{op} not allowed. Not in {implemented_ops}.'
                ops_list.append(implemented_ops[op])

            elif callable(op):
                ops_list.append(op)

            else:
                raise ValueError(f'{op} is neither a string nor a callable.')
    
    elif isinstance(augmentations, dict):
        for op in augmentations:
            if op in implemented_ops:
                assert len(augmentations[op]) == 2, f'{op} value must be of length 2.'
                ops_list.append(ft.partial(implemented_ops[op], param_range=augmentations[op]))

            elif callable(augmentations[op]):
                ops_list.append(augmentations[op])
            
            else:
                raise ValueError(f'{op} value is neither a list of parameter range for implemented operations nor a callable.')
    else:
        raise ValueError('augmentations is neither a list nor a dictionary.')

    return ops_list


@tf.function
def get_target_tf(marker_file, target_shape, dim_resolution):
    def get_target_wrap(marker_file, target_shape, dim_resolution):
        marker_file = Path(marker_file.decode())
        blobs = get_target(
            marker_file,
            target_shape=target_shape, 
            default_radius=3.5, 
            dim_resolution=dim_resolution,
        ) # FIXME! non configurable args!!
        return blobs.astype(np.float32)

    target = tf.numpy_function(get_target_wrap, [marker_file, target_shape, dim_resolution], tf.float32)
    return target


@tf.function
def get_input_tf(input_file):
    def get_input_wrap(input_file):
        input_file = Path(input_file.decode())
        input_image = InputFile(input_file).whole().astype(np.float32)
        return input_image

    input = tf.numpy_function(get_input_wrap, [input_file], tf.float32)
    return input


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
        shape = tf.shape(input_image)

        logger.info(f'creating blobs from {marker_path}')
        shape_T = [shape[d] for d in (2, 1, 0)]
        dim_resolution_T = [dim_resolution[d] for d in (2, 1, 0)]
        blobs = get_target_tf(marker_path, shape_T, dim_resolution_T)
        blobs = tf.transpose(blobs, perm=(2, 1, 0))
        return input_image, blobs

    def __new__(cls, tiff_list, marker_list, batch_size, output_shape, dim_resolution=1.0, augmentations=None, augmentations_prob=0.5):
        if isinstance(dim_resolution, (float, int)):
            dim_resolution = (dim_resolution) * 3
        
        # load inputs and targets from paths
        data = tf.data.Dataset.from_tensor_slices((tiff_list, marker_list))
        data = data.map(lambda x, y: cls.parse_imgs(x, y, dim_resolution), num_parallel_calls=20)
        data = data.cache().shuffle(len(marker_list), reshuffle_each_iteration=True)

        # crop inputs and targets
        data = data.map(lambda x, y: random_crop_tf(x, y, output_shape))

        # do augmentations
        if augmentations is not None:
            ops_list = get_op_list(augmentations)
            data = data.map(lambda x, y: augment(x, y, ops_list, augmentations_prob))
        
        # add channel dimension
        data = data.map(lambda x, y: (tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)))

        return data.batch(batch_size)


if __name__ == '__main__':
    import os
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # marker_dir = '/mnt/NASone/curzio/Data/I48/Gray_matter/GT_files/Train'
    # img_dir = '/mnt/NASone/curzio/Data/I48/Gray_matter/Tiff_files/Train'

    # marker_files = [f'{marker_dir}/{f}' for f in os.listdir(marker_dir)]
    # img_files = [f'{img_dir}/{f}' for f in os.listdir(img_dir)]

    data_dir = '/mnt/NASone/curzio/Data/SST'
    marker_files = [f'{data_dir}/GT_files/Train/{fname}' for fname in os.listdir(f'{data_dir}/GT_files/Train')]
    img_files = [f'{data_dir}/Tiff_files/Train/{fname}' for fname in os.listdir(f'{data_dir}/Tiff_files/Train')]
    
    ordered_tiff_list = []
    for f in marker_files:
        fname = Path(f).with_suffix('').name
        tiff_file = [f for f in map(lambda f: Path(f), img_files) if f.name == fname]
        ordered_tiff_list.append(str(tiff_file[0]))

    data = TrainingDataset(
        ordered_tiff_list[:10],
        marker_files[:10],
        batch_size=2,
        output_shape=[50, 100, 100],
        dim_resolution=[2.0, 0.65, 0.65],
        augmentations={
            'gamma':[0.5, 2.0],
            'rotation':[-180, 180],
            'zoom':[1.3, 1.5],
            'brightness':[-50, 100],
            'noise':[0.001, 0.05],
            'myfunc': lambda x, y: (x - tf.reduce_min(x), y),
            },
        augmentations_prob=[0.3, 0.3, 0.0, 0.0, 0.3],
        )

    for e in range(3):
        for x, y in data:
            print()
            print(x.shape, tf.reduce_min(x, axis=[1, 2, 3, 4]).numpy())
            print(y.shape, tf.reduce_min(y, axis=[1, 2, 3, 4]).numpy())