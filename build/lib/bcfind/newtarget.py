import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import gaussian_filter

from matplotlib import animation, pyplot as plt
from matplotlib.ticker import MultipleLocator
from colorama import Fore as FG

ZANISOTROPY = 3
DEBUGGING = True


def printd(*args):
    if DEBUGGING:
        print(*args)


def iround(val):
    return int(round(val))


def get_target(fname, default_radius=4.5, safe_factor=5.0):
    # Radius to be used when cells are sufficiently far away
    # Radius should be never larger than distance to the nearest neighbor divided by this quantity
    assert fname.endswith("marker")
    printd(FG.RED, "Processing file", fname, FG.RESET)
    df = pd.read_csv(open(fname, "r"))[["#x", " y", " z"]]
    X = df.values
    # D = cdist(X,X,'chebyshev')
    # D = cdist(X,X,'euclidean')
    D = cdist(
        X,
        X,
        lambda u, v: np.sqrt(
            (u[0] - v[0]) ** 2
            + (u[1] - v[1]) ** 2
            + (ZANISOTROPY * u[2] - ZANISOTROPY * v[2]) ** 2
        ),
    )
    D = D + 1e30 * np.eye(D.shape[0])  # get rid of diagonal
    a = np.unravel_index(D.argmin(), D.shape)  # pair with shortest distance
    radii = (
        {}
    )  # For each xyz triplet, the target radius that can be used without making overlaps
    for c in X:
        radii[tuple(map(iround, c))] = default_radius
    while D[a] < safe_factor * default_radius:
        printd(fname, a, X[a[0]], X[a[1]], D[a])
        for c in [X[a[0]], X[a[1]]]:
            real_r = min(
                radii[tuple(map(iround, c))], D[a] / safe_factor
            )  # reduce radius if cells are too close
            radii[tuple(map(iround, c))] = (
                int(real_r * 10) / 10.0
            )  # Quantize at 0.1 resolution to limit the number of distinct radii
        # get the next smallest distance..
        D[a] = 1e30
        D[a[1], a[0]] = 1e30
        a = np.unravel_index(D.argmin(), D.shape)
    target = np.zeros((480, 480, 160))  # FIXME: constants..
    printd("Looping on", len(set(radii.values())), "values of the radius")
    for r in set(
        radii.values()
    ):  # this could be slow especially if there are too many distinct radii
        centers = [
            c for c in radii if np.abs(radii[c] - r) < 1e-10
        ]  # cells that have this r
        if r < default_radius:
            printd(FG.GREEN, r, centers, FG.RESET)
        else:
            printd(FG.CYAN, r, centers[:8], "...", FG.RESET)
            # continue
        component = np.zeros((480, 480, 160))
        for c in centers:
            component[c[0], c[1], c[2]] = 1
        sigma = max(1.5, r)
        component = gaussian_filter(
            component,
            (sigma, sigma, sigma / ZANISOTROPY),
            truncate=sigma / 2,
            mode="constant",
        )
        component = component / component.max()
        target = target + component
        printd(
            "  ->Created component for radius",
            r,
            "with sigma",
            sigma,
            "for a total of",
            len(centers),
            "cells",
        )
    target = target / target.max()
    # Making a movie for inspection, alternatively save as an npz file for training..
    # NOTE: this may require transposing since here the target tensor has axes ordered as x,y,z
    # target = (255*target).astype(np.uint8) # use 8 bit for the video
    # make_video(target,fname+'.mp4')
    return target
