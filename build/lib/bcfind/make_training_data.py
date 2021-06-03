import os
import h5py
import argparse
import numpy as np
import pandas as pd
import functools as ft
import scipy.ndimage.filters as sp_filt
import scipy.spatial.distance as sp_dist

from skimage import io
from colorama import Fore as FG

from bcfind.config_manager import Configuration
from bcfind.utils import preprocessing, iround


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="make_training_data.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to .yaml file containing the needed configuration settings",
    )
    return parser.parse_args()


def get_target(
    file_path,
    target_shape,
    default_radius=3.5,
    safe_factor=3.0,
    dim_resolution=1,
    downscale_factors=None,
):
    # Radius to be used when cells are sufficiently far away
    # Radius should be never larger than distance to the nearest neighbor divided by this quantity
    # default_radius *= dim_resolution

    df = pd.read_csv(open(file_path, "r"))[["#x", " y", " z"]]
    df = df.dropna(0)

    X = df.to_numpy()
    if downscale_factors is not None:
        X *= downscale_factors

    # X_um = X * dim_resolution

    if X.shape[0] == 0:
        print(
            FG.RED,
            f"Marker file {file_path} is empty. Black target returned.",
            FG.RESET,
        )
        return np.zeros(target_shape)

    else:
        print(FG.GREEN, "Processing file", file_path, FG.RESET)
        # D = cdist(X,X,'chebyshev')
        D = sp_dist.cdist(X, X, "euclidean")
        D = D + 1e30 * np.eye(D.shape[0])  # get rid of diagonal

        a = np.unravel_index(D.argmin(), D.shape)  # pair with shortest distance

        radii = {}
        # For each xyz triplet, the target radius that can be used without making overlaps
        for c in X:
            radii[tuple(map(iround, c))] = default_radius

        while D[a] < safe_factor * default_radius:
            for c in [X[a[0]], X[a[1]]]:
                # reduce radius if cells are too close
                real_r = min(radii[tuple(map(iround, c))], D[a] / safe_factor)
                # Quantize at 0.1 resolution to limit the number of distinct radii
                radii[tuple(map(iround, c))] = int(real_r * 10) / 10.0

            # get the next smallest distance..
            D[a[0], a[1]] = 1e30
            D[a[1], a[0]] = 1e30

            a = np.unravel_index(D.argmin(), D.shape)

        target = np.zeros(target_shape)
        print("Looping on", len(set(radii.values())), "values of the radius")

        # this could be slow especially if there are too many distinct radii
        for r in set(radii.values()):
            # cells that have this r
            centers = [c for c in radii if np.abs(radii[c] - r) < 1e-10]

            component = np.zeros(target_shape)
            for c in centers:
                c = list(c)
                if c[0] == component.shape[0]:
                    c[0] = c[0] - 1
                if c[1] == component.shape[1]:
                    c[1] = c[1] - 1
                if c[2] == component.shape[2]:
                    c[2] = c[2] - 1

                component[c[0], c[1], c[2]] = 1

            sigma = max(1.5, r / np.min(dim_resolution))
            dim_sigma = sigma / (dim_resolution / np.min(dim_resolution))

            component = sp_filt.gaussian_filter(
                component, dim_sigma, truncate=2.5, mode="constant"
            )
            component = component / component.max()

            target = target + component

            print(
                "  ->Created component for radius",
                r,
                "with sigma",
                sigma,
                "for a total of",
                len(centers),
                "cells",
            )

        target = target / target.max()

    return target


def make_data(
    train_tif_dir,
    train_gt_dir,
    preprocessing_fun,
    data_shape,
    dim_resolution,
    downscale_factors,
    x_file,
    y_file,
):
    outdir = os.path.dirname(x_file)
    assert outdir == os.path.dirname(
        y_file
    ), "Base directory for x and y files must be equal."
    os.makedirs(outdir, exist_ok=True)

    tifnames = [
        f
        for f in os.listdir(train_tif_dir)
        if f.endswith(".tif") or f.endswith(".tiff")
    ]
    n = len(tifnames)

    fx = h5py.File(x_file, "w")
    fy = h5py.File(y_file, "w")

    print("Creating x dataset")
    fx.create_dataset(name="x", data=np.zeros((n, *data_shape), dtype=np.uint8))
    print("Creating y dataset")
    fy.create_dataset(name="y", data=np.zeros((n, *data_shape), dtype=np.uint8))

    print("Filling in..")
    all_fnames = []
    for i, fname in enumerate(tifnames):
        all_fnames.append(fname)
        tif_path = f"{train_tif_dir}/{fname}"
        marker_path = f"{train_gt_dir}/{fname}.marker"

        image = io.imread(tif_path)
        image = preprocessing_fun(image)

        try:
            target = get_target(
                marker_path,
                data_shape,
                default_radius=3.5,  # FIXME: not yet configurable
                safe_factor=3.5,  # FIXME: not yet configurable
                dim_resolution=dim_resolution,
                downscale_factors=downscale_factors,
            )
        except FileNotFoundError:
            print(
                f"File .marker for {fname} not found. Assumed without neurons. Black target returned."
            )
            target = np.zeros(data_shape)

        fx["x"][i, ...] = (image * 255).astype(np.uint8)
        fy["y"][i, ...] = (target * 255).astype(np.uint8)

    fx.close()
    fy.close()

    with h5py.File(x_file, "r") as fx:
        X = fx["x"][()]
    with h5py.File(y_file, "r") as fy:
        Y = fy["y"][()]

    all_fnames = np.array(all_fnames)
    np.save(f"{outdir}/file_names.npy", all_fnames)
    return X, Y, all_fnames


def main():
    args = parse_args()
    conf = Configuration(args.config)

    x_file = f"{conf.exp.train_data_dir}/X_train.h5"
    y_file = f"{conf.exp.train_data_dir}/Y_train.h5"

    preprocessing_fun = ft.partial(
        preprocessing,
        transpose=conf.preproc.transpose,
        flip_axis=conf.preproc.flip_axis,
        clip_threshold=conf.preproc.clip_threshold,
        gamma_correction=conf.preproc.gamma_correction,
        downscale_factors=conf.preproc.downscale_factors,
        pad_output_shape=conf.data.data_shape,
    )

    make_data(
        conf.data.train_tif_dir,
        conf.data.train_gt_dir,
        preprocessing_fun,
        conf.data.data_shape,
        conf.data.dim_resolution,
        conf.preproc.downscale_factors,
        x_file,
        y_file,
    )


if __name__ == "__main__":
    main()
