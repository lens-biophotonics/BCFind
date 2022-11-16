import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor

from zetastitcher import InputFile

from bcfind.data.artificial_targets import get_target


def vaa3d_to_numpy(marker_path):
    df = pd.read_csv(open(str(marker_path), "r"))[['#x', ' y', ' z']]
    df = df.dropna(axis=0)
    return df.to_numpy()


def slicer_to_numpy(marker_path):
    with open(marker_path, 'r') as f:
        markers = json.load(f)
    X = []
    control_points = markers['markups'][0]['controlPoints']
    for cp in control_points:
        X.append(cp['position'])
    return np.array(X)


def get_gt_as_numpy(marker_path):
    if isinstance(marker_path, Path):
        suffix = marker_path.suffix
    else:
        _, suffix = os.path.splitext(marker_path)
    
    if suffix == '.marker':
        gt = vaa3d_to_numpy(marker_path)
    elif suffix == '.json':
        gt = slicer_to_numpy(marker_path)
    else:
        raise ValueError('marker_path is incompatible with known formats: Vaa3d (.marker) or 3DSlicer (.json).')
    return gt[:, [2, 1, 0]] # transpose axis from [x, y, z] to [z, y, x]


def bit_clip(x, bit=14):
    clip = 2 ** bit
    return np.where(x > clip, clip, x)


def windsor_clip(x, quantile=99):
    q = np.quantile(x, quantile)
    return np.where(x > q, q, x)


def auto_clip(x):
    lof = LocalOutlierFactor(n_neighbors=50, contamination=0.2)
    inl_pred = lof.fit_predict(x.reshape(-1, 1))
    in_max = np.max(x.flatten()[inl_pred > 0.5])
    new_x = np.where(x > in_max, in_max, x)
    return new_x.astype(np.float32)


def preprocess(x):
    new_x = windsor_clip(x, quantile=99)
    # new_x = bit_clip(new_x, bit=14)
    # new_x = auto_clip(new_x)
    return new_x / new_x.max()


@tf.function(reduce_retracing=True)
def get_input_tf(input_file):
    def get_input_wrap(input_file):
        input_file = Path(input_file.decode())
        input_image = InputFile(input_file).whole()
        input_image = input_image.astype(np.float32)
        return preprocess(input_image)

    input = tf.numpy_function(get_input_wrap, [input_file], tf.float32)
    return input


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