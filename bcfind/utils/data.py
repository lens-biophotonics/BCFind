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
    df = pd.read_csv(open(str(marker_path), "r"))[["#x", " y", " z"]]
    df = df.dropna(axis=0)
    return df.to_numpy()


def slicer_to_numpy(marker_path):
    with open(marker_path, "r") as f:
        markers = json.load(f)
    X = []
    control_points = markers["markups"][0]["controlPoints"]
    for cp in control_points:
        X.append(cp["position"])
    return np.array(X)


def get_gt_as_numpy(marker_path):
    if isinstance(marker_path, Path):
        suffix = marker_path.suffix
    else:
        _, suffix = os.path.splitext(marker_path)

    if suffix == ".marker":
        gt = vaa3d_to_numpy(marker_path)
    elif suffix == ".json":
        gt = slicer_to_numpy(marker_path)
    else:
        raise ValueError(
            "marker_path is incompatible with known formats: Vaa3d (.marker) or 3DSlicer (.json)."
        )
    return gt[:, [2, 1, 0]]  # transpose axis from [x, y, z] to [z, y, x]


def auto_clip(x):
    lof = LocalOutlierFactor(n_neighbors=50, contamination=0.2)
    inl_pred = lof.fit_predict(x.reshape(-1, 1))
    in_max = np.max(x.flatten()[inl_pred > 0.5])
    new_x = np.where(x > in_max, in_max, x)
    return new_x.astype(np.float32)


def get_preprocess_func(
    clip="bit",
    clip_value=15,
    center=None,
    center_value=0,
    scale="bit",
    scale_value=15,
    slice_p=None,
):
    if clip not in ["constant", "bit", "quantile", "auto", "none", None]:
        raise ValueError(
            f'clip argument must be one of "constant", "bit", "quantile", "auto", "none" or None. Got {clip}'
        )
    if center not in ["constant", "min", "mean", "none", None]:
        raise ValueError(
            f'center argument must be on of "constant", "min", "mean", "none" or None. Got {center}'
        )
    if scale not in ["constant", "bit", "max", "std", "none", None]:
        raise ValueError(
            f'scale argument must be on of "constant", "bit", "max", "std", "none" or None. Got {scale}'
        )

    if clip == "constant":
        clip_fun = lambda x: np.where(x > clip_value, clip_value, x)
    elif clip == "bit":
        clip_fun = lambda x: np.where(x > 2**clip_value, 2**clip_value, x)
    elif clip == "quantile":
        clip_fun = lambda x: np.where(
            x > np.quantile(x, clip_value), np.quantile(x, clip_value), x
        )
    elif clip == "auto":
        clip_fun = auto_clip
    elif clip in [None, "none"]:
        clip_fun = lambda x: x

    if center == "constant":
        center_fun = lambda x: x - center_value
    elif center == "min":
        center_fun = lambda x: x - x.min()
    elif center == "mean":
        center_fun = lambda x: x - x.mean()
    elif center in [None, "none"]:
        center_fun = lambda x: x

    if scale == "constant":
        scale_fun = lambda x: x / scale_value
    elif scale == "bit":
        scale_fun = lambda x: x / 2**scale_value
    elif scale == "max":
        scale_fun = lambda x: x / x.max()
    elif scale == "std":
        scale_fun = lambda x: x / x.std()
    elif scale in [None, "none"]:
        scale_fun = lambda x: x

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

        # ATTN!: hard coded slice norm
        try:
            if kwargs["slice_p"] is not None:
                sl = int(input_file.name.split("_")[1])
                sl_p = int(kwargs["slice_p"])

                print(f"Using percentiles_{sl_p:02}.json for slice norm")
                lomin, himax = json.load(
                    open(
                        f"/home/Data/I48_slab{sl:02}/NeuN638/percentiles_{sl_p:02}.json"
                    )
                ).values()
                input_image = np.where(input_image > himax, himax, input_image)
                input_image = input_image - lomin
                input_image = np.where(input_image < 0, 0, input_image)
                input_image = input_image / (himax - lomin)
        except:
            pass

        preprocess_func = get_preprocess_func(**kwargs)
        return preprocess_func(input_image)

    input = tf.numpy_function(get_input_wrap, [input_file], tf.float32)
    return input
