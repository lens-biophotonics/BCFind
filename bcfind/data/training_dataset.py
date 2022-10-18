import os
import lmdb
import pickle
import logging
import numpy as np
import tensorflow as tf
import concurrent.futures as cf

from pathlib import Path

from zetastitcher import InputFile
from bcfind.data.artificial_targets import get_target
from bcfind.data.augmentation import *


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@tf.function(reduce_retracing=True)
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


def normalize_tf(x):
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    new_x = (x - x_min) / (x_max - x_min)
    return new_x


@tf.function(reduce_retracing=True)
def get_input_tf(input_file):
    def get_input_wrap(input_file):
        input_file = Path(input_file.decode())
        input_image = InputFile(input_file).whole()
        return input_image.astype(np.float32)

    input = tf.numpy_function(get_input_wrap, [input_file], tf.float32)
    input = normalize_tf(input)
    return input

from sklearn.neighbors import LocalOutlierFactor
@tf.function()
def auto_clip_tf(x):
    def auto_clip_wrap(x):
        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.2)
        inl_pred = lof.fit_predict(x.reshape(-1, 1))
        in_max = np.max(x.flatten()[inl_pred > 0.8])
        in_min = np.max(x.flatten()[inl_pred > 0.8])
        x[x>in_max] = in_max
        x[x<in_min] = in_min
        return x.astype(np.float32)

    x = tf.numpy_function(auto_clip_wrap, [x], tf.float32)
    return x


class TrainingDataset(tf.keras.utils.Sequence):
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

        logger.info(f'creating blobs from {marker_path}')
        blobs = get_target_tf(marker_path, tf.shape(input_image), dim_resolution)

        xy = tf.concat([tf.expand_dims(input_image, 0), tf.expand_dims(blobs, 0)], axis=0)
        return tf.ensure_shape(xy, (2, None, None, None))

    def __new__(
        cls, 
        tiff_list, 
        marker_list, 
        batch_size, 
        dim_resolution=1.0, 
        output_shape=None, 
        augmentations=None, 
        augmentations_prob=0.5,
        use_lmdb_data = False,
        ):
        if not use_lmdb_data:
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
        else:
            obj = super(TrainingDataset, cls).__new__(cls)
            return obj

    def __init__(
        self, 
        tiff_list, 
        marker_list, 
        batch_size, 
        dim_resolution=1.0, 
        output_shape=None, 
        augmentations=None, 
        augmentations_prob=0.5,
        use_lmdb_data=False,
        n_workers=10,
        ):

        self.tiff_list = tiff_list
        self.marker_list = marker_list
        self.batch_size = batch_size
        self.dim_resolution = dim_resolution
        self.output_shape = output_shape
        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.use_lmdb_data = use_lmdb_data
        self.n_workers = n_workers
        self.n = len(self.tiff_list)

        if isinstance(dim_resolution, (float, int)):
            dim_resolution = [dim_resolution] * 3
        
        if use_lmdb_data:
            self.lmdb_path = os.path.join('/', *self.tiff_list[0].split('/')[:-3], 'Train_lmdb')
            
            # NOTE: hardcoded data shape for lmdb size!
            nbytes = np.prod((300, 500, 500)) * 4 # 4 bytes for float32: 1 byte for uint8
            self.map_size = 2*self.n*nbytes*10
            self.lmdb_env = lmdb.open(self.lmdb_path, map_size=self.map_size, max_dbs=2)
            self.inputs = self.lmdb_env.open_db('Inputs'.encode())
            self.targets = self.lmdb_env.open_db('Targets'.encode())
            
            if not os.path.isdir(self.lmdb_path):
                self.create_lmdb()
            else:
                print(f'Found lmdb data at {self.lmdb_path}. Data will be taken from there')
            
    def create_lmdb(self,):
        print('Creating lmdb data')
        with self.lmdb_env.begin(write=True) as txn:
            i = 0
            for tiff_file, marker_file in zip(self.tiff_list, self.marker_list):
                print(f'Writing {i+1}/{len(self.tiff_list)} input-target pair to lmdb')
                i += 1
                
                x = get_input_tf(tiff_file)
                y = get_target_tf(marker_file, tf.shape(x), self.dim_resolution)

                fname = tiff_file.split('/')[-1]
                txn.put(key=fname.encode(), value=pickle.dumps(x), db=self.inputs)
                txn.put(key=fname.encode(), value=pickle.dumps(y), db=self.targets)

        print('Closing lmdb')
        self.lmdb_env.close()
    
    def on_epoch_end(self,):
        print('Epoch ended. Shuffling files.')
        tif_mark = list(zip(self.tiff_list, self.marker_list))
        np.random.shuffle(tif_mark)
        self.tiff_list, self.marker_list = zip(*tif_mark)
        
    def __len__(self,):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):
        files = self.tiff_list[idx * self.batch_size : (idx + 1) * self.batch_size]
        fnames = [f.split('/')[-1] for f in files]

        with self.lmdb_env.begin() as txn:
            x_batch, y_batch = [], []
            for f in fnames:
                x = txn.get(key=f.encode(), db=self.inputs)
                x = pickle.loads(x)

                y = txn.get(key=f.encode(), db=self.targets)
                y = pickle.loads(y)

                if self.output_shape:
                    xy = tf.concat([x[tf.newaxis, ...], y[tf.newaxis, ...]], axis=0)
                    xy = random_crop_tf(xy, self.output_shape)
                    x, y = tf.unstack(xy)
                if self.augmentations:
                    xy = tf.concat([x[tf.newaxis, ...], y[tf.newaxis, ...]], axis=0)
                    xy = augment(xy, self.augmentations, self.augmentations_prob)
                    x, y = tf.unstack(xy)
                
                x_batch.append(x[tf.newaxis, ..., tf.newaxis])
                y_batch.append(y[tf.newaxis, ..., tf.newaxis])
            
        return tf.concat(x_batch, axis=0), tf.concat(y_batch, axis=0)

    def getitem(self, i):
        for item in self[i]:
            yield item

    def __iter__(self):
        with cf.ThreadPoolExecutor(self.n_workers) as pool:
            futures = [pool.submit(self.getitem, i) for i in range(len(self))]
            for future in cf.as_completed(futures):
                yield future.result()
                

    
                