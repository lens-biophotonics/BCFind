import logging
from multiprocessing.sharedctypes import Value
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


class TrainingDataset:
    """ Training dataset flow for 3D gray images with cell locations as targets.
    Args:
        tiff_list (list):
        marker_list (list):
        output_shape (list):
        augmentations (optional, list, dict):
        augmentations_prob (optional, float, list):

    Returns:
        tensorflow.datasets.Data: a tensorflow dataset.

    Yields:
        tensorflow.Tensor: batch of inputs and targets.
    """
    @staticmethod
    def _img_generator(tiff_list, marker_list):
        for marker_file in map(lambda x: Path(x.decode()), marker_list):
            fname = marker_file.with_suffix('').name

            input_file = [f for f in map(lambda x: Path(x.decode()), tiff_list) if f.name == fname]
            
            if len(input_file) > 1:
                raise ValueError(f'{len(input_file)} tiff files have the same name.')
            elif len(input_file) == 0:
                raise ValueError(f'Tiff file {fname} not found!')

            logger.info(f'loading {input_file[0]}')
            input_image = InputFile(input_file[0]).whole().astype(np.float32)        

            logger.info(f'creating blobs from {marker_file}')
            blobs = get_target(marker_file, [input_image.shape[d] for d in [2, 1, 0]], default_radius=1.5)

            blobs = blobs.transpose((2, 1, 0))
            yield input_image, blobs

    def __new__(cls, tiff_list, marker_list, batch_size, output_shape, augmentations=None, augmentations_prob=0.5):        
        def _my_gen():
            for el in crop_gen:
                yield el[0], el[1]

        img_gen = tf.data.Dataset.from_generator(
            cls._img_generator,
            output_signature=tf.TensorSpec(None, tf.float32),
            args=(tiff_list, marker_list))

        random_crop_wrap = ft.partial(random_crop, zoom_range=None, target_shape=output_shape)

        crop_gen = img_gen \
            .cache() \
            .shuffle(len(marker_list), reshuffle_each_iteration=True) \
            .map(lambda el: tf.numpy_function(random_crop_wrap, [el[0], el[1]], (tf.float32, tf.float32)))

        if augmentations is not None:
            ops_list = get_op_list(augmentations)

            block_gen = tf.data.Dataset.from_generator(_my_gen, output_signature=(
                    (tf.TensorSpec((tuple(output_shape) + (1,)), tf.float32),) * 2))
            block_gen = block_gen.map(lambda x, y: (augment(x, ops_list, augmentations_prob), y))
            return block_gen.batch(batch_size)
        
        else:
            return crop_gen.batch(batch_size)


if __name__ == '__main__':
    import os
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    marker_dir = '/mnt/NASone/curzio/Data/I48/Gray_matter/GT_files/Train'
    img_dir = '/mnt/NASone/curzio/Data/I48/Gray_matter/Tiff_files/Train'

    marker_files = [f'{marker_dir}/{f}' for f in os.listdir(marker_dir)]
    img_files = [f'{img_dir}/{f}' for f in os.listdir(img_dir)]

    data = TrainingDataset(
        img_files, 
        marker_files, 
        batch_size=2, 
        output_shape=[50, 100, 100], 
        augmentations=None,#('gamma', 'noise', lambda x: x - tf.reduce_min(x)),
        augmentations_prob=1.0
        )

    for x, y in data:
        print()
        print(x.shape, tf.reduce_min(x, axis=[1, 2, 3, 4]).numpy())