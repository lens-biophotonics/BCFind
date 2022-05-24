import numpy as np
import pandas as pd
import scipy.ndimage.filters as sp_filt
import scipy.spatial.distance as sp_dist

from colorama import Fore as FG

from bcfind.utils import iround


def get_target(
    marker_path,
    target_shape,
    default_radius=3.5,
    safe_factor=3.5,
    dim_resolution=1,
    downscale_factors=None,
    verbose=False,
):
    # Radius to be used when cells are sufficiently far away
    # Radius should be never larger than distance to the nearest neighbor divided by this quantity
    # default_radius *= dim_resolution
    df = pd.read_csv(open(str(marker_path), "r"))
    df = df.iloc[:, :3]
    df = df.dropna(0)
    df = df[(df.iloc[:, 0]<target_shape[0]) & (df.iloc[:, 1]<target_shape[1]) & (df.iloc[:, 2]<target_shape[2])]
    df = df[(df.iloc[:, 0]>0) & (df.iloc[:, 1]>0) & (df.iloc[:, 2]>0)]

    X = df.to_numpy()
    if downscale_factors is not None:
        X *= downscale_factors

    # X_um = X * dim_resolution

    if X.shape[0] == 0:
        if verbose:
            print(
                FG.RED,
                f"Marker file {marker_path} is empty. Black target returned.",
                FG.RESET,
            )
        return np.zeros(target_shape)

    else:
        if verbose:
            print(FG.GREEN, "Processing file", marker_path, FG.RESET)

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
        if verbose:
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

            sigma = max(1, r / np.min(dim_resolution))
            dim_sigma = sigma / (dim_resolution / np.min(dim_resolution))

            component = sp_filt.gaussian_filter(
                component, dim_sigma, truncate=2, mode="constant"
            )
            component = component / component.max()

            target = target + component

            if verbose:
                print(
                    f"---> Created component for radius {r}"
                    f" with sigma {dim_sigma}"
                    f" for a total of {len(centers)} cells"
                )

        target = target / target.max()

    return target