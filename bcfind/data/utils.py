import os
import json
import numpy as np
import pandas as pd
import functools as ft
import tensorflow as tf

from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor

from zetastitcher import InputFile


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


def get_preprocess_func(clip='quantile', clip_value=99, center='min', center_value=None, scale='max', scale_value=None):
    if clip not in ['constant', 'quantile', 'bit', 'auto', 'none', None]:
        raise ValueError(f'clip argument must be on of "constant", "quantile", "bit" or "auto", "none" or None. Got {clip}')
    if center not in ['min', 'constant', 'mean', 'none', None]:
        raise ValueError(f'center argument must be on of "min", "constant", "mean", "none" or None. Got {center}')
    if scale not in ['max', 'constant', 'std']:
        raise ValueError(f'scale argument must be on of "max", "constant" or "std". Got {scale}')
    
    if clip == 'constant':
        clip_fun = lambda x: np.where(x > clip_value, clip_value, x)
    elif clip == 'quantile':
        clip_fun = ft.partial(windsor_clip, quantile=clip_value)
    elif clip == 'bit':
        clip_fun = ft.partial(bit_clip, bit=clip_value)
    elif clip == 'auto':
        clip_fun = auto_clip
    elif clip in [None, 'none']:
        clip_fun = lambda x: x
    
    if center == 'min':
        center_fun = lambda x: x - x.min()
    elif center == 'mean':
        center_fun = lambda x: x - x.mean()
    elif center == 'constant':
        center_fun = lambda x: x - center_value
    elif center in [None, 'none']:
        center_fun = lambda x: x

    if scale == 'max':
        scale_fun = lambda x: x / x.max()
    elif scale == 'constant':
        scale_fun = lambda x: x / scale_value
    elif scale == 'std':
        scale_fun = lambda x: x / x.std()

    def func(x):
        new_x = clip_fun(x)
        new_x = center_fun(new_x)
        new_x = scale_fun(new_x)
        return new_x
    
    return func


@tf.function(reduce_retracing=True)
def get_input_tf(input_file, **kwargs):
    def get_input_wrap(input_file):
        input_file = Path(input_file.decode())
        input_image = InputFile(input_file).whole()
        input_image = input_image.astype(np.float32)
        preprocess_func = get_preprocess_func(**kwargs)
        return preprocess_func(input_image)

    input = tf.numpy_function(get_input_wrap, [input_file], tf.float32)
    return input