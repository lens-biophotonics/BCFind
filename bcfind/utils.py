import numpy as np
import skimage.transform as sk_trans
from skimage import io


def iround(val):
    return int(round(val))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def metrics(df):
    try:
        prec = np.sum(df.TP) / np.sum(df.TP + df.FP)
    except ZeroDivisionError:
        prec = 1.0
    try:
        rec = np.sum(df.TP) / np.sum(df.TP + df.FN)
    except ZeroDivisionError:
        rec = 1.0
    try:
        f1 = 2.0 * prec * rec / (prec + rec)
    except ZeroDivisionError:
        f1 = 0.0
    return {"prec": prec, "rec": rec, "f1": f1}


def get_substack(
    file_path,
    data_shape,
    transpose=None,
    flip_axis=None,
    clip_threshold=None,
    gamma_correction=None,
    downscale_factors=None,
):
    im = io.imread(file_path)

    # Preprocessing
    if transpose is not None:
        im = np.transpose(im, transpose)
    if flip_axis is not None:
        im = np.flip(im, axis=flip_axis)
    if clip_threshold is not None:
        im[np.where(im > clip_threshold)] = clip_threshold
    if gamma_correction is not None:
        im = np.power(im / im.max(), gamma_correction)
    if downscale_factors is not None:
        im = sk_trans.rescale(im, downscale_factors, anti_aliasing=False)

    im_shape = im.shape

    substack = np.zeros(data_shape)  # paddle substacks with zeros
    substack[: im_shape[0], : im_shape[1], : im_shape[2]] = im

    return substack
