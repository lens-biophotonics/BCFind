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


def pad(img, output_shape):
    padded_img = np.zeros(output_shape)
    insert_here = tuple(slice(0, img.shape[dim]) for dim in range(len(img.shape)))
    padded_img[insert_here] = img
    return padded_img


def preprocessing(
    img,
    transpose=None,
    flip_axis=None,
    clip_threshold=None,
    gamma_correction=None,
    downscale_factors=None,
    pad_output_shape=None,
):
    if transpose is not None:
        img = np.transpose(img, transpose)
    if flip_axis is not None:
        img = np.flip(img, axis=flip_axis)
    if clip_threshold is not None:
        img[np.where(img > clip_threshold)] = clip_threshold
    if gamma_correction is not None:
        img = np.power(img / img.max(), gamma_correction)
    if downscale_factors is not None:
        img = sk_trans.rescale(img, downscale_factors, anti_aliasing=False)
    if pad_output_shape is not None:
        img = pad(img, pad_output_shape)

    return img / img.max()
