import os
import h5py
import argparse
import numpy as np
import pandas as pd
import scipy.ndimage.filters as sp_filt
import scipy.spatial.distance as sp_dist
from colorama import Fore as FG

from config_manager import Configuration
from utils import get_substack, iround


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
            print(file_path, a, X[a[0]], X[a[1]], D[a])

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

            if r < default_radius:
                print(FG.BLUE, r, centers, FG.RESET)
            else:
                print(FG.CYAN, r, centers[:8], "...", FG.RESET)
                # continue

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


def main(args):
    conf = Configuration(args.config)

    os.makedirs(conf.data.files_h5_dir, exist_ok=True)

    tifnames = [f for f in os.listdir(conf.data.train_tif_dir)]
    n = len(tifnames)

    fx = h5py.File(f"{conf.data.files_h5_dir}/X_train.h5", "w")
    fy = h5py.File(f"{conf.data.files_h5_dir}/Y_train.h5", "w")

    print("Creating x dataset")
    fx.create_dataset(
        name="x", data=np.zeros((n, *conf.data.data_shape), dtype=np.uint8)
    )
    print("Creating y dataset")
    fy.create_dataset(
        name="y", data=np.zeros((n, *conf.data.data_shape), dtype=np.uint8)
    )

    print("Filling in..")
    all_fnames = []
    for i, fname in enumerate(tifnames):
        print(f"Processing {fname}")

        all_fnames.append(fname)
        tif_path = f"{conf.data.train_tif_dir}/{fname}"
        marker_path = f"{conf.data.train_gt_dir}/{fname}.marker"

        image = get_substack(
            tif_path,
            conf.data.data_shape,
            transpose=conf.preproc.transpose,
            flip_axis=conf.preproc.flip_axis,
            clip_threshold=conf.preproc.clip_threshold,
            gamma_correction=conf.preproc.gamma_correction,
            downscale_factors=conf.preproc.downscale_factors,
        )
        try:
            target = get_target(
                marker_path,
                conf.data.data_shape,
                default_radius=3.5,
                safe_factor=3.5,
                dim_resolution=conf.data.dim_resolution,
                downscale_factors=conf.preproc.downscale_factors,
            )
        except FileNotFoundError:
            print(
                f"File .marker for {fname} not found. Assumed without neurons. Black target returned."
            )
            target = np.zeros(conf.data.input_shape)

        fx["x"][i, :, :, :] = (image * 255).astype(np.uint8)
        fy["y"][i, :, :, :] = (target * 255).astype(np.uint8)

    fx.close()
    fy.close()

    all_fnames = np.array(all_fnames)
    np.save(f"{conf.data.files_h5_dir}/file_names.npy", all_fnames)


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="make_training_data.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument("config", type=str, help="configuration file")
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
